#!/usr/bin/env python3
"""Check that every concrete codebase-note pointer resolves to a unique note.

Source comments point at `docs/codebase_notes.md` by note ID instead of carrying
the longform rationale inline. That indirection is only trustworthy if the
pointers resolve, so this fails the build on a dangling reference -- the failure
mode being a note renamed or deleted while comments still point at it.

Notes with no inbound reference are reported too, but do not fail: a note whose
code was deleted should be marked superseded rather than silently orphaned.
Numeric and legacy uppercase symbolic suffixes are supported; malformed IDs
fail. The documented ID template and family wildcards are placeholders.

Usage:  python3 scripts/check_note_refs.py [--verbose]
"""

import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(REPO, "docs", "codebase_notes.md")
# Read the entire token so malformed suffixes cannot pass as valid prefixes.
TOKEN_RE = re.compile(r"(?<![\w-])CN-[\w<>*\-]+")
ID_RE = re.compile(r"CN-[A-Z0-9]+-(?:[0-9]+|[A-Z][A-Z0-9]*)\Z")
PLACEHOLDER_RE = re.compile(r"CN-(?:\*|[A-Z0-9]+-\*)\Z")


def note_tokens(text):
    for match in TOKEN_RE.finditer(text):
        token = match.group()
        if token == "CN-<AREA>-<n>" or PLACEHOLDER_RE.fullmatch(token):
            continue
        yield token, text.count("\n", 0, match.start()) + 1


SUFFIXES = (".cu", ".cuh", ".h", ".hh", ".cpp", ".cc", ".inl", ".md", ".py",
            ".toml", ".cmake", ".sh")
SKIP_DIRS = {".git", "build", "third_party", "__pycache__", "temp", "data"}


def is_source_file(path):
    parts = path.replace("\\", "/").split("/")
    if any(part in SKIP_DIRS or part.startswith("build") for part in parts[:-1]):
        return False
    return path.endswith(SUFFIXES) or parts[-1] == "CMakeLists.txt"


def tracked_files():
    """Prefer Git's tracked and untracked source list, but fall back to a walk.

    `git ls-files` is the accurate list, but it fails in CI containers that trip
    git's "dubious ownership" check on a checkout owned by another uid. Falling
    back to a walk keeps the check running there rather than failing for a
    reason that has nothing to do with note references.
    """
    try:
        out = subprocess.run(["git", "-C", REPO, "ls-files", "--cached",
                              "--others", "--exclude-standard"],
                             capture_output=True, text=True, check=True)
        files = [f for f in out.stdout.splitlines() if is_source_file(f)]
        if files:
            return files
    except (subprocess.CalledProcessError, OSError):
        pass

    files = []
    for root, dirs, names in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                   and not d.startswith("build")]
        for name in names:
            rel = os.path.relpath(os.path.join(root, name), REPO)
            if is_source_file(rel):
                files.append(rel)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(NOTES):
        print("FAIL missing required docs/codebase_notes.md")
        return 1

    try:
        with open(NOTES, encoding="utf-8") as notes:
            text = notes.read()
    except (OSError, UnicodeError) as error:
        print(f"FAIL cannot read docs/codebase_notes.md: {error}")
        return 1
    defined, failures = set(), []
    for heading in re.finditer(r"^##\s+(CN-[\w<>*\-]+)", text, re.M):
        nid = heading.group(1)
        line = text.count("\n", 0, heading.start()) + 1
        if not ID_RE.fullmatch(nid):
            failures.append(f"docs/codebase_notes.md:{line}: malformed note ID {nid}")
        elif nid in defined:
            failures.append(f"docs/codebase_notes.md:{line}: duplicate note ID {nid}")
        else:
            defined.add(nid)

    refs, dangling = {}, []
    for rel in tracked_files():
        path = os.path.join(REPO, rel)
        try:
            body = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for nid, line in note_tokens(body):
            if not ID_RE.fullmatch(nid):
                failures.append(f"{rel}:{line}: malformed note ID {nid}")
                continue
            if rel == "docs/codebase_notes.md" and re.match(
                    rf"^##\s+{re.escape(nid)}\b", body.splitlines()[line - 1]):
                continue
            refs.setdefault(nid, []).append(rel)
            if nid not in defined:
                dangling.append((rel, nid))

    for rel, nid in dangling:
        print(f"FAIL {rel}: references {nid}, which is not defined in "
              f"docs/codebase_notes.md")
    for failure in failures:
        print(f"FAIL {failure}")

    orphans = sorted(defined - set(refs))
    if orphans and args.verbose:
        print("\nNotes with no inbound reference (mark superseded if the code "
              "is gone):")
        for nid in orphans:
            print(f"  {nid}")

    if args.verbose:
        print(f"\n{len(defined)} notes, {sum(len(v) for v in refs.values())} "
              f"references, {len(orphans)} unreferenced")
    if dangling or failures:
        print(f"\n{len(dangling)} dangling reference(s), {len(failures)} validation problem(s)")
        return 1
    print(f"OK: {sum(len(v) for v in refs.values())} note references all resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
