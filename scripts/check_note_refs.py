#!/usr/bin/env python3
"""Check that every `CN-<AREA>-<n>` pointer in the tree resolves to a note.

Source comments point at `docs/codebase_notes.md` by note ID instead of carrying
the longform rationale inline. That indirection is only trustworthy if the
pointers resolve, so this fails the build on a dangling reference -- the failure
mode being a note renamed or deleted while comments still point at it.

Notes with no inbound reference are reported too, but do not fail: a note whose
code was deleted should be marked superseded rather than silently orphaned.

Usage:  python3 scripts/check_note_refs.py [--verbose]
"""

import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(REPO, "docs", "codebase_notes.md")
ID_RE = re.compile(r"CN-[A-Z0-9]+-\d+")


SUFFIXES = (".cu", ".cuh", ".h", ".hh", ".cpp", ".cc", ".inl", ".md", ".py",
            ".toml")
SKIP_DIRS = {".git", "build", "third_party", "__pycache__", "temp", "data"}


def tracked_files():
    """Prefer git, but fall back to a walk.

    `git ls-files` is the accurate list, but it fails in CI containers that trip
    git's "dubious ownership" check on a checkout owned by another uid. Falling
    back to a walk keeps the check running there rather than failing for a
    reason that has nothing to do with note references.
    """
    try:
        out = subprocess.run(["git", "-C", REPO, "ls-files"],
                             capture_output=True, text=True, check=True)
        files = [f for f in out.stdout.splitlines() if f.endswith(SUFFIXES)]
        if files:
            return files
    except (subprocess.CalledProcessError, OSError):
        pass

    files = []
    for root, dirs, names in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                   and not d.startswith("build")]
        for name in names:
            if name.endswith(SUFFIXES):
                files.append(os.path.relpath(os.path.join(root, name), REPO))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(NOTES):
        print("no docs/codebase_notes.md; nothing to check")
        return 0

    text = open(NOTES, encoding="utf-8").read()
    defined = set(re.findall(r"^##\s+(CN-[A-Z0-9]+-\d+)", text, re.M))

    refs, dangling = {}, []
    for rel in tracked_files():
        if os.path.basename(rel) == "codebase_notes.md":
            continue
        path = os.path.join(REPO, rel)
        try:
            body = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for nid in ID_RE.findall(body):
            refs.setdefault(nid, []).append(rel)
            if nid not in defined:
                dangling.append((rel, nid))

    for rel, nid in dangling:
        print(f"FAIL {rel}: references {nid}, which is not defined in "
              f"docs/codebase_notes.md")

    orphans = sorted(defined - set(refs))
    if orphans and args.verbose:
        print("\nNotes with no inbound reference (mark superseded if the code "
              "is gone):")
        for nid in orphans:
            print(f"  {nid}")

    if args.verbose:
        print(f"\n{len(defined)} notes, {sum(len(v) for v in refs.values())} "
              f"references, {len(orphans)} unreferenced")
    if dangling:
        print(f"\n{len(dangling)} dangling note reference(s)")
        return 1
    print(f"OK: {sum(len(v) for v in refs.values())} note references all resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
