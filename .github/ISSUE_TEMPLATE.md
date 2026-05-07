---
name: Bug report
about: Report a bug or unexpected behavior
---

## Description

A clear description of the bug and what you expected to happen instead.

## Environment

**OS** (`cat /etc/os-release` or `uname -a`):

**GPU** (`nvidia-smi -L`):

**CUDA version** (`nvcc --version`):

**Compiler** (`gcc --version` or `clang --version`):

**CMake version** (`cmake --version`):

**FZGPUModules version / commit** (`git log -n 1 --oneline`):

**CMake configuration** (`cmake -L <build_dir>` or the preset and flags you used):

## Steps to Reproduce

1. 
2. 
3. 

**Expected behavior:**

**Actual behavior:**

A minimal self-contained reproducer (C++ snippet, test name, or script) is very helpful.

## Additional Context

Logs, stack traces, sanitizer output, or anything else relevant. If you ran under
AddressSanitizer or Compute Sanitizer, include that output here.
