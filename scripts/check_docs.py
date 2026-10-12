#!/usr/bin/env python3
"""Run documentation and stage integration checks without building the library."""

import argparse
from pathlib import Path
import subprocess
import sys


CHECKS = (
    ("TOML examples and presets", "check_doc_toml.py", True),
    ("C++ examples", "check_doc_cpp.py", True),
    ("Codebase note references", "check_note_refs.py", True),
    ("Stage integration", "check_stage_integration.py", False),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    failed = []
    for label, script, supports_verbose in CHECKS:
        print(f"\nChecking {label}...", flush=True)
        command = [sys.executable, str(scripts / script)]
        if args.verbose and supports_verbose:
            command.append("--verbose")
        result = subprocess.run(command, cwd=scripts.parent)
        if result.returncode:
            failed.append(label)
    if failed:
        print("\nFAIL: " + ", ".join(failed))
        return 1
    print("\nOK: all four documentation/integration checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
