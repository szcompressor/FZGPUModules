#!/usr/bin/env python3
"""Validate every TOML example embedded in the docs against the real loader.

The two things that actually rot in a TOML example are the stage `type` string
and the parameter key names, and both are knowable from `src/pipeline/config.cpp`
without running anything:

  * `kStageRegistry` is the authoritative list of accepted `type` strings.
  * each `addXxxStage()` reads its parameters through `optInt/optStr/optDbl/
    optBool(t, "key", ...)`, so the literals in that function body are exactly
    the keys that stage accepts.

Deriving both from the source means this check cannot disagree with the code the
way a hand-maintained list does. Fragments (a bare `[[stage]]` block with no
`[pipeline]` table) are checked for type/keys only; whole-config examples are
additionally checked for structural sanity.

Usage:  python3 scripts/check_doc_toml.py [--verbose]
Exit status is non-zero if any example would be rejected by the loader.
"""

import argparse
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_CPP = os.path.join(REPO, "src", "pipeline", "config.cpp")

# Keys consumed by the generic [[stage]] / [pipeline] plumbing rather than by any
# one stage's loader, so they never appear in an addXxxStage() body.
STRUCTURAL_STAGE_KEYS = {"name", "type", "inputs", "from", "port"}
PIPELINE_KEYS = {
    "input_size", "dims", "memory_strategy", "pool_multiplier", "num_streams",
    "primary_source",
}
# Reserved `from` value meaning "bind directly to the pipeline's raw input"
# (Pipeline::bindExternalInput()) rather than another declared stage's output
# -- never itself a declared stage name, so it's exempted from the dangling-
# reference check below.
EXTERNAL_INPUT_SENTINEL = "__external__"


def parse_registry(src):
    """type_name -> loader function name, from kStageRegistry."""
    table = re.search(r"kStageRegistry\[\]\s*=\s*\{(.*?)\n\};", src, re.S)
    if not table:
        sys.exit("could not locate kStageRegistry in config.cpp")
    out = {}
    for type_name, load_fn in re.findall(
        r'\{\s*"([A-Za-z0-9_]+)"\s*,\s*StageType::[A-Z0-9_]+\s*,\s*(\w+)\s*,', table.group(1)
    ):
        out[type_name] = load_fn
    return out


def parse_loader_keys(src, fn_name):
    """The set of TOML keys an addXxxStage() function reads."""
    m = re.search(r"static\s+Stage\*\s+%s\s*\([^)]*\)\s*\{" % re.escape(fn_name), src)
    if not m:
        return None
    # Walk braces from the opening one to find the function body.
    i = src.index("{", m.end() - 1)
    depth, j = 0, i
    while j < len(src):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    body = src[i:j]
    return set(re.findall(r'\bopt(?:Int|Str|Dbl|Bool)\s*\(\s*t\s*,\s*"([^"]+)"', body))


def extract_toml_blocks(path):
    text = open(path, encoding="utf-8").read()
    blocks = []
    # Track line numbers so failures point somewhere useful.
    for m in re.finditer(r"```toml\n(.*?)```", text, re.S):
        line = text.count("\n", 0, m.start()) + 1
        blocks.append((line, m.group(1)))
    return blocks


def strip_comments(block):
    return "\n".join(l.split("#", 1)[0] for l in block.splitlines())


def check_block(block, registry, keys_by_type, errors, where):
    body = strip_comments(block)

    # Split into [[stage]] sections; anything before the first is [pipeline]-ish.
    sections = re.split(r"^\s*\[\[stage\]\]\s*$", body, flags=re.M)
    head, stages = sections[0], sections[1:]

    for key in re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", head, re.M):
        if key not in PIPELINE_KEYS:
            errors.append(f"{where}: unknown [pipeline] key '{key}'")

    declared = set()
    for sec in stages:
        tm = re.search(r'^\s*type\s*=\s*"([^"]+)"', sec, re.M)
        if not tm:
            errors.append(f"{where}: [[stage]] block with no `type`")
            continue
        stype = tm.group(1)
        if stype not in registry:
            errors.append(
                f"{where}: type \"{stype}\" is not an accepted stage type "
                f"(not in kStageRegistry)"
            )
            continue
        nm = re.search(r'^\s*name\s*=\s*"([^"]+)"', sec, re.M)
        if nm:
            declared.add(nm.group(1))

        accepted = keys_by_type.get(stype)
        if accepted is None:
            continue
        for key in re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", sec, re.M):
            if key in STRUCTURAL_STAGE_KEYS or key in accepted:
                continue
            # `segments` is read by the Merge loader through a different helper.
            if stype == "Merge" and key == "segments":
                continue
            errors.append(
                f"{where}: stage type \"{stype}\" has no key '{key}' "
                f"(accepts: {', '.join(sorted(accepted)) or 'none'})"
            )

    # Only whole-config examples can be checked for dangling references; a bare
    # fragment legitimately refers to stages defined elsewhere in the page.
    if head.strip():
        for ref in re.findall(r'from\s*=\s*"([^"]+)"', body):
            if ref not in declared and ref != EXTERNAL_INPUT_SENTINEL:
                errors.append(f"{where}: inputs reference undefined stage \"{ref}\"")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    src = open(CONFIG_CPP, encoding="utf-8").read()
    registry = parse_registry(src)
    keys_by_type = {t: parse_loader_keys(src, fn) for t, fn in registry.items()}

    files = sorted(glob.glob(os.path.join(REPO, "docs", "**", "*.md"), recursive=True))
    files += [os.path.join(REPO, f) for f in ("README.md", "AGENTS.md", "CONTRIBUTING.md")]

    errors, n = [], 0
    for path in files:
        if not os.path.exists(path):
            continue
        for line, block in extract_toml_blocks(path):
            n += 1
            rel = os.path.relpath(path, REPO)
            check_block(block, registry, keys_by_type, errors, f"{rel}:{line}")

    # The shipped presets rot exactly the same way, and they are what users copy.
    presets = sorted(glob.glob(os.path.join(REPO, "examples", "presets", "*.toml")))
    for path in presets:
        n += 1
        rel = os.path.relpath(path, REPO)
        check_block(open(path, encoding="utf-8").read(), registry, keys_by_type,
                    errors, f"{rel}:1")

    if args.verbose:
        print(f"{len(registry)} stage types, {n} TOML examples/presets checked "
              f"({len(presets)} presets)")
    for e in errors:
        print(f"FAIL {e}")
    if errors:
        print(f"\n{len(errors)} problem(s) in documented TOML examples")
        return 1
    print(f"OK: {n} TOML examples/presets valid against kStageRegistry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
