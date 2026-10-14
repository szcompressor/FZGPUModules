#!/usr/bin/env python3
"""Build an isolated source snapshot, delete it, and test a relocated CUDA install.

Requires a CUDA GPU/toolchain. Unlike a build-tree smoke test, old source-dependent
NVRTC code fails here: its configured source and build directories no longer exist.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cuda-architectures", default="native")
    parser.add_argument("--compute-sanitizer", action="store_true",
                        help="wrap the installed consumer in Compute Sanitizer memcheck")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.compute_sanitizer and shutil.which("compute-sanitizer") is None:
        parser.error("--compute-sanitizer requires compute-sanitizer on PATH")
    with tempfile.TemporaryDirectory(prefix="fzgm-installed-",
                                     dir=os.environ.get("RUNNER_TEMP")) as temporary:
        root = Path(temporary)
        source, build, prefix = root / "source", root / "build", root / "install"
        source.mkdir()
        # Explicit source directories avoid copying datasets, results and local builds.
        for name in ("include", "modules", "src", "cmake", "scripts", "third_party",
                     "examples"):
            shutil.copytree(REPO / name, source / name,
                            ignore=shutil.ignore_patterns(".git", "__pycache__"))
        for name in ("CMakeLists.txt", "LICENSE", "THIRD_PARTY.md"):
            shutil.copy2(REPO / name, source / name)
        fixture = root / "consumer"
        shutil.copytree(REPO / "tests" / "installed_specialization", fixture)
        run(["cmake", "-S", str(source), "-B", str(build),
             "-DCMAKE_BUILD_TYPE=Release", "-DFZGMOD_BACKEND=CUDA",
             "-DBUILD_TESTING=OFF", "-DBUILD_EXAMPLES=OFF", "-DENABLE_NVTX=OFF",
             "-DBUILD_CLI=OFF", "-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_architectures])
        run(["cmake", "--build", str(build), "--parallel", str(args.jobs)])
        run(["cmake", "--install", str(build), "--prefix", str(prefix)])
        consumer_build = root / "consumer-build"
        run(["cmake", "-S", str(fixture), "-B", str(consumer_build),
             "-DCMAKE_PREFIX_PATH=" + str(prefix)])
        run(["cmake", "--build", str(consumer_build), "--parallel", str(args.jobs)])
        (prefix / "bin").mkdir(exist_ok=True)
        shutil.copy2(consumer_build / "installed_specialization", prefix / "bin")
        shutil.rmtree(source)
        shutil.rmtree(build)
        shutil.rmtree(fixture)
        shutil.rmtree(consumer_build)
        relocated = root / "relocated"
        prefix.rename(relocated)
        output = root / "output"
        output.mkdir()
        environment = os.environ.copy()
        # No process-global policy override may defeat the fixture's Off/Auto checks.
        for name in ("FZ_SPECIALIZE", "FZ_FUSION"):
            environment.pop(name, None)
        libraries = [str(relocated / "lib"), str(relocated / "lib64")]
        if environment.get("LD_LIBRARY_PATH"):
            libraries.append(environment["LD_LIBRARY_PATH"])
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
        command = [str(relocated / "bin" / "installed_specialization")]
        if args.compute_sanitizer:
            command = ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "1"] + command
        for mode in ("write", "read"):
            run(command + [mode, str(output)], cwd=output, env=environment)
        print("Installed specialization passed with source/build removed and install relocated.")


if __name__ == "__main__":
    main()
