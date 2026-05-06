# Contributing

Thanks for your interest in improving FZGPUModules. Contributions of all kinds are welcome: bug reports, documentation updates, examples, tests, and code changes.

## Ways to contribute

- Report bugs with a minimal reproducer and environment details.
- Improve documentation or examples.
- Add tests for new features or regressions.
- Implement fixes or features discussed in issues.

## Build and test

This project uses CMake presets.

```bash
cmake --preset release
cmake --build build/release -j$(nproc)
```

To build examples or tests:

```bash
cmake --preset release -DBUILD_EXAMPLES=ON
cmake --preset release -DBUILD_TESTS=ON
cmake --build build/release -j$(nproc)
```

To run tests after building with `-DBUILD_TESTS=ON`:

```bash
ctest --test-dir build/release --output-on-failure
```

## Style and conventions

- Keep changes focused and add tests where practical.
- Update documentation and examples when behavior changes.
- For code changes, add a concise entry to `CHANGELOG.md` under the appropriate `[Unreleased]` section (not needed for docs-only changes).
- Use public headers in examples and user-facing code (include `fzgpumodules.h`, do not include internal headers).

## Submitting changes

- Open a PR against the default branch with a clear description and rationale.
- Link relevant issues and include reproduction steps for bug fixes.
- Mention any build flags or environment specifics required to validate the change.
