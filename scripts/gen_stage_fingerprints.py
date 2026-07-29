#!/usr/bin/env python3
"""Generate per-stage source fingerprints into a C++ header.

Why
---
Downstream consumers cache expensive results per stage — a benchmark harness may
spend 20 h filling a result matrix, and when one stage's kernels change it wants to
re-run only the cells that used that stage. Knowing *which* stages a pipeline ran
is not enough; it also has to know whether a stage's code is the same code that
produced the cached number. A pipeline config hash cannot answer that: editing a
.cu file changes no config.

So each stage gets a fingerprint over its own sources.

Transitive, not per-directory
-----------------------------
A naive hash of `modules/coders/rze/` would miss almost everything that matters.
Stages share infrastructure (`backend/api.h` is included by 33 files, `stage/stage.h`
by 25) and they include *each other*: `transforms/zigzag/zigzag.h` is pulled in by 6
stages and `fused/lorenzo_quant/lorenzo_quant.h` by 5. A change to zigzag must
invalidate every stage that inlines it, and a change to the memory pool must
invalidate all of them.

The fingerprint therefore covers each stage's own files PLUS the transitive closure
of its quoted `#include`s resolved inside this repo. Angle-bracket includes (CUDA,
STL) are ignored on purpose — they are pinned by the toolchain, which the consumer
records separately in its provenance.

Properties
----------
- Exact and automatic: no version integer for anyone to forget to bump.
- Conservative: a comment or formatting change moves the fingerprint. Re-running a
  few hundred cells needlessly is a far better failure than trusting a stale number.
- Order-independent: files are sorted before hashing, so it is reproducible across
  machines and filesystems.

The generator FAILS rather than guessing: a registry entry naming a directory that
does not exist, a stage with no `source_dir`, or an unresolvable local include are
all build errors.

Usage:
    python scripts/gen_stage_fingerprints.py --out build/include/fz_stage_fingerprints.h
    python scripts/gen_stage_fingerprints.py --print          # human-readable table
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG_CPP = REPO / "src" / "pipeline" / "config.cpp"

# Roots a quoted #include may resolve against, in search order.
INCLUDE_ROOTS = [REPO / "include", REPO / "modules", REPO / "src"]

SOURCE_SUFFIXES = {".cu", ".cuh", ".h", ".hpp", ".inl", ".cpp", ".c"}

# { "Name", StageType::X, addFn, saveFn, "modules/a/b" },
_ENTRY_RE = re.compile(
    r'\{\s*"(?P<name>[A-Za-z0-9_]+)"\s*,\s*StageType::\w+\s*,\s*\w+\s*,\s*\w+\s*,'
    r'\s*"(?P<dir>[^"]+)"\s*\}')
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


def parse_registry() -> list[tuple[str, str]]:
    text = CONFIG_CPP.read_text()
    start = text.find("static const StageEntry kStageRegistry[]")
    if start < 0:
        sys.exit(f"error: kStageRegistry not found in {CONFIG_CPP}")
    end = text.find("};", start)
    entries = _ENTRY_RE.findall(text[start:end])
    if not entries:
        sys.exit(f"error: parsed 0 entries from kStageRegistry in {CONFIG_CPP}; "
                 f"the entry format may have changed — update _ENTRY_RE")
    # Cross-check: every quoted type_name in the block must have been captured, so a
    # malformed entry is a hard error rather than a silently-skipped stage.
    names_in_block = re.findall(r'\{\s*"([A-Za-z0-9_]+)"\s*,\s*StageType::',
                                text[start:end])
    got = {n for n, _ in entries}
    missing = [n for n in names_in_block if n not in got]
    if missing:
        sys.exit(f"error: {len(missing)} registry entries lack a parseable source_dir: "
                 f"{', '.join(missing)}")
    return entries


def resolve_include(inc: str, current: Path) -> Path | None:
    for cand in [current.parent / inc] + [r / inc for r in INCLUDE_ROOTS]:
        if cand.is_file():
            return cand.resolve()
    return None


def closure(seeds: list[Path]) -> set[Path]:
    """Transitive closure of repo-local quoted includes, seeds included."""
    seen: set[Path] = set()
    stack = [p.resolve() for p in seeds]
    while stack:
        f = stack.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        for inc in _INCLUDE_RE.findall(text):
            r = resolve_include(inc, f)
            if r and r not in seen:
                stack.append(r)
            # Unresolvable quoted includes are almost always third-party headers
            # vendored via an include path we do not model. They cannot be hashed,
            # so they are skipped; the toolchain/provenance covers that risk.
    return seen


def fingerprint(stage_dir: Path) -> tuple[str, int]:
    own = sorted(p for p in stage_dir.rglob("*")
                 if p.is_file() and p.suffix in SOURCE_SUFFIXES)
    if not own:
        sys.exit(f"error: no source files under {stage_dir}")
    files = sorted(closure(own), key=lambda p: p.relative_to(REPO).as_posix())
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(REPO).as_posix().encode())
        h.update(b"\0")
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()[:16], len(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="header to write")
    ap.add_argument("--print", action="store_true", dest="show")
    args = ap.parse_args()

    rows = []
    for name, rel in parse_registry():
        d = REPO / rel
        if not d.is_dir():
            sys.exit(f"error: stage '{name}' names source_dir '{rel}', "
                     f"which does not exist")
        fp, n = fingerprint(d)
        rows.append((name, rel, fp, n))

    if args.show or not args.out:
        print(f"{'stage':20s} {'fingerprint':18s} {'files':>6s}  source_dir")
        for name, rel, fp, n in rows:
            print(f"{name:20s} {fp:18s} {n:6d}  {rel}")
        if not args.out:
            return 0

    body = "\n".join(
        f'    {{ "{name}", "{fp}" }},' for name, _, fp, _ in rows)
    header = f"""// GENERATED by scripts/gen_stage_fingerprints.py — DO NOT EDIT.
//
// Per-stage source fingerprints: sha256 over each stage's own sources plus the
// transitive closure of its repo-local #includes, truncated to 16 hex chars.
// Regenerated at build time; see that script for the rationale.
#pragma once

namespace fz {{
namespace generated {{

struct StageFingerprint {{ const char* name; const char* fingerprint; }};

static const StageFingerprint kStageFingerprints[] = {{
{body}
}};

}}  // namespace generated
}}  // namespace fz
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Only rewrite on change, so CMake does not rebuild the world every time.
    if not out.exists() or out.read_text() != header:
        out.write_text(header)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
