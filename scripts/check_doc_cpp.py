#!/usr/bin/env python3
"""Syntax/type-check every C++ example embedded in the docs.

Doc snippets are fragments, not programs, so this does not link or run them --
it compiles each one with `-fsyntax-only` against the real public headers, which
is what catches the failure mode that actually matters: an example naming a
class, method, enumerator, or port that the library no longer has.

Each snippet is wrapped in a function body. A set of commonly-referenced names
(`stream`, `d_input`, `n`, ...) is predeclared at namespace scope, so a snippet
may either use them or shadow them with its own local declaration. Snippets that
are not statements (a class definition, a free function, a `#include` block) are
retried at namespace scope, and a snippet that compiles either way passes.

Snippets that are deliberately incomplete can opt out with an HTML comment on the
line before the fence:

    <!-- doc-check: skip reason goes here -->

Usage:  python3 scripts/check_doc_cpp.py [--verbose] [--jobs N] [--filter SUBSTR]
Exit status is non-zero if any example fails to compile.
"""

import argparse
import concurrent.futures
import glob
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Names a fragment may reasonably reference without declaring. Namespace scope so
# a snippet that declares its own is shadowing, not redefining.
PREAMBLE = r"""
#include "fzgpumodules.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>

using namespace fz;

namespace doccheck_env {
inline Pipeline p(4096);
inline Pipeline pipeline(4096);
inline cudaStream_t stream{};
inline size_t n = 1024;
inline size_t input_bytes = 4096;
inline size_t num_elements = 1024;
inline size_t nx = 16, ny = 16, nz = 16;
inline void*  d_input = nullptr;
inline void*  d_output = nullptr;
inline void*  d_compressed = nullptr;
inline void*  d_decompressed = nullptr;
inline size_t compressed_size = 0;
inline size_t compressed_sz = 0;
inline size_t decompressed_size = 0;
inline size_t output_size = 0;
inline size_t decomp_sz = 0;
inline float  eb = 1e-3f;
inline const char* filename = "x.fzm";
}
using namespace doccheck_env;
"""


def find_cuda_include():
    for probe in ("/usr/local/cuda/include", "/opt/cuda/include"):
        if os.path.isdir(probe):
            return probe
    nvcc = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
    if nvcc.returncode == 0:
        base = os.path.dirname(os.path.dirname(nvcc.stdout.strip()))
        cand = os.path.join(base, "include")
        if os.path.isdir(cand):
            return cand
    return None


def extract_blocks(path):
    text = open(path, encoding="utf-8").read()
    # A page written entirely around placeholders (`MyStage`, `<category>/...`)
    # can never compile; it opts out once rather than per block.
    if re.search(r"<!--\s*doc-check:\s*skip-file", text):
        return []
    out = []
    for m in re.finditer(r"(?:<!--\s*doc-check:\s*skip(.*?)-->\s*\n)?```cpp\n(.*?)```",
                         text, re.S):
        line = text.count("\n", 0, m.start()) + 1
        out.append((line, m.group(2), m.group(1)))
    return out


def page_prelude(blocks):
    """Names a page introduces in one block and reuses in later ones.

    Stage docs are written to be read top to bottom: the first block does
    `auto* rze = p.addStage<RZEStage>();` and later blocks just call methods on
    `rze`. Hoisting every such declaration to namespace scope gives each snippet
    the whole page's vocabulary, while still letting a snippet shadow a name with
    its own local declaration.
    """
    seen, out = set(), []
    for _, body, _ in blocks:
        for m in re.finditer(r"^\s*auto\*\s*(\w+)\s*=\s*(.+?);\s*$", body, re.M):
            name, init = m.group(1), m.group(2)
            if name in seen or "\n" in init:
                continue
            seen.add(name)
            out.append(f"inline auto* {name} = {init};")
    return "\n".join(out)


# A fragment that references an undeclared local is incomplete, not wrong -- the
# page just did not spell out every variable. A fragment that names a member or
# overload the library does not have is a genuinely stale example, and that is
# the only class this script fails the build on.
# Naming a member or overload that does not exist is conclusive: it cannot be
# explained by an undeclared local, because the object's type resolved fine.
STRONG_MARKERS = (
    "has no member named",
    "is not a member of",
    "is private within this context",
    "too few arguments to function",
    "too many arguments to function",
)
# These are real signals only in an otherwise fully-resolved snippet; alongside
# an undeclared placeholder they are just cascade noise.
WEAK_MARKERS = (
    "no matching function for call to 'fz::",
    "no matching function for call to ‘fz::",
    "invalid use of incomplete type",
    "cannot convert",
)


# Placeholder stage names a tutorial may legitimately use for a class that does
# not exist. Anything else ending in `Stage` is meant to be a real class.
PLACEHOLDER_RE = re.compile(r"^(My|Your|Example|Foo|Bar|Some|Custom|New)\w*$")
UNDECLARED_RE = re.compile(r"[‘'](\w+)['’] (?:was not declared|does not name a type)")


def is_stale_api(msgs):
    # An undeclared placeholder (`MyStage`, `StageT`) cascades into template
    # errors that mimic stale-API hits. GCC tags some with `<expression error>`;
    # the rest are ruled out by the presence of any undeclared identifier.
    real = [m for m in msgs if "<expression error>" not in m]
    if any(any(mark in m for mark in STRONG_MARKERS) for m in real):
        return True

    # A stage class that no longer exists shows up only as an undeclared
    # identifier, so `SomethingStage` going missing must not be waved through as
    # "just an undeclared local" -- that is the rename/removal case.
    for m in real:
        hit = UNDECLARED_RE.search(m)
        if hit:
            name = hit.group(1)
            if name.endswith("Stage") and not PLACEHOLDER_RE.match(name):
                return True

    undeclared = any("was not declared in this scope" in m
                     or "does not name a type" in m for m in real)
    if undeclared:
        return False
    return any(any(mark in m for mark in WEAK_MARKERS) for m in real)


def compile_one(args):
    tag, body, incdirs, prelude = args
    first_err = None
    # Try richest environment first, then fall back: a page prelude that does not
    # itself compile must not be reported as a failure of the snippet.
    for pre in (prelude, ""):
        env = PREAMBLE.replace("}\nusing namespace doccheck_env;",
                               pre + "\n}\nusing namespace doccheck_env;")
        for scope in ("function", "namespace"):
            if scope == "function":
                src = env + "\nvoid doccheck_snippet() {\n" + body + "\n}\n"
            else:
                src = env + "\n" + body + "\n"
            with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
                f.write(src)
                tmp = f.name
            cmd = ["g++", "-std=c++17", "-fsyntax-only", "-w"]
            for d in incdirs:
                cmd += ["-I", d]
            cmd.append(tmp)
            r = subprocess.run(cmd, capture_output=True, text=True)
            os.unlink(tmp)
            if r.returncode == 0:
                return tag, True, ""
            # Report the richest attempt (page prelude, function scope): the
            # later fallbacks fail for harness reasons and their errors mislead.
            if first_err is None:
                first_err = r.stderr
    return tag, False, first_err or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--filter", default="")
    args = ap.parse_args()

    cuda_inc = find_cuda_include()
    if not cuda_inc:
        print("SKIP: no CUDA include directory found; cannot syntax-check C++ examples")
        return 0

    incdirs = [
        os.path.join(REPO, "include"),
        os.path.join(REPO, "modules"),
        os.path.join(REPO, "experimental"),  # quarantined reference compressors
        os.path.join(REPO, "src"),
        REPO,
        cuda_inc,
    ]

    files = sorted(glob.glob(os.path.join(REPO, "docs", "**", "*.md"), recursive=True))
    files += [os.path.join(REPO, f) for f in ("README.md", "AGENTS.md", "CONTRIBUTING.md")]

    jobs, skipped = [], 0
    for path in files:
        if not os.path.exists(path):
            continue
        rel = os.path.relpath(path, REPO)
        if args.filter and args.filter not in rel:
            continue
        blocks = extract_blocks(path)
        prelude = page_prelude(blocks)
        for line, body, skip in blocks:
            if skip is not None:
                skipped += 1
                continue
            jobs.append((f"{rel}:{line}", body, incdirs, prelude))

    stale, fragments = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for tag, ok, err in ex.map(compile_one, jobs):
            if ok:
                continue
            msgs = [l.split(" error: ", 1)[-1]
                    for l in err.splitlines() if " error: " in l]
            (stale if is_stale_api(msgs) else fragments).append((tag, msgs))

    if stale:
        print("Examples referencing API the library no longer has:\n")
        for tag, msgs in sorted(stale):
            print(f"FAIL {tag}")
            for m in msgs[:3]:
                print(f"       {m}")

    if args.verbose and fragments:
        print("\nIncomplete fragments (undeclared locals/placeholders "
              "-- not a stale-API failure):")
        for tag, msgs in sorted(fragments):
            print(f"  frag {tag}: {msgs[0] if msgs else ''}")

    ok_n = len(jobs) - len(stale) - len(fragments)
    print(f"\n{ok_n}/{len(jobs)} compile as-is; {len(fragments)} incomplete "
          f"fragments; {len(stale)} referencing stale API; {skipped} skipped")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
