#!/usr/bin/env bash
# verify_stage.sh — one answer to "is this new stage done?"
#
# Usage: .claude/skills/fzgm-add-stage/verify_stage.sh [--static] <lower_name> [build_dir]
#   --static   : only the file/integration checks (seconds, no build or GPU)
#   lower_name : snake_case module name as created by scripts/new_stage.sh (e.g. xor_delta)
#   build_dir  : default build/release (configured with -DBUILD_TESTING=ON)
#
# Environment:
#   FZ_SKIP_SANITIZER=1   skip the Compute Sanitizer pass (e.g. no compute-sanitizer)
#   FZ_GPU_WRAPPER="..."  prefix for GPU commands (e.g. a GPU lock: "python -m benchkit lock --")
#   JOBS=N                build parallelism (default: nproc)
#
# Exit 0 and "RESULT: PASS" only if every check passes.

set -uo pipefail

STATIC=0
if [[ "${1:-}" == --static ]]; then STATIC=1; shift; fi
NAME="${1:-}"
if [[ -z "$NAME" || "$NAME" == -h || "$NAME" == --help ]]; then
    sed -n '2,14p' "$0"; exit 2
fi
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT" || exit 2
BUILD="${2:-build/release}"
JOBS="${JOBS:-$(nproc)}"
read -r -a WRAP <<< "${FZ_GPU_WRAPPER:-}"

FAILS=()
pass() { printf '  PASS  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; FAILS+=("$*"); }
warn() { printf '  WARN  %s\n' "$*"; }
section() { printf '\n== %s\n' "$*"; }

# ── Locate the stage ──────────────────────────────────────────────────────────
MODULE_DIR=$(find modules -mindepth 2 -maxdepth 2 -type d -name "$NAME" | head -1)
if [[ -z "$MODULE_DIR" ]]; then
    echo "No module directory modules/<category>/$NAME — run scripts/new_stage.sh first."; exit 2
fi
CATEGORY=$(basename "$(dirname "$MODULE_DIR")")
HEADER="$MODULE_DIR/${NAME}_stage.h"
SOURCE="$MODULE_DIR/${NAME}_stage.cu"
TEST_SRC="tests/stages/test_${NAME}.cpp"
TEST_BIN_NAME="test_${NAME}"
UPPER=$(grep -oP 'StageType::\K[A-Z0-9_]+' "$SOURCE" 2>/dev/null | head -1)
echo "Stage: $MODULE_DIR (category $CATEGORY, StageType::${UPPER:-?})"

# ── Static checks (no build needed) ───────────────────────────────────────────
section "Integration and docs checks"
if python3 scripts/check_docs.py > /tmp/fz_verify_docs_$$.log 2>&1; then
    pass "scripts/check_docs.py (includes check_stage_integration.py)"
else
    fail "scripts/check_docs.py — see output below"; sed 's/^/        /' /tmp/fz_verify_docs_$$.log | grep -E 'FAIL|ERROR|error|problem' | head -20
fi
rm -f /tmp/fz_verify_docs_$$.log

for f in "$HEADER" "$SOURCE" "$TEST_SRC" "$MODULE_DIR/CMakeLists.txt"; do
    [[ -f "$f" ]] || { fail "missing $f"; continue; }
    n=$(grep -c 'TODO' "$f")
    if [[ "$n" -gt 0 ]]; then fail "$f has $n TODO(s) left from the scaffold"; else pass "no TODOs in $f"; fi
done

[[ -n "$UPPER" ]] || fail "could not find the StageType used in $SOURCE"
if [[ -n "$UPPER" ]]; then
    grep -q "StageType::${UPPER}\b" tests/pipeline/test_stage_registry.cpp \
        && pass "StageType::$UPPER listed in tests/pipeline/test_stage_registry.cpp" \
        || fail "add StageType::$UPPER to shipped[] in tests/pipeline/test_stage_registry.cpp"
    grep -q "StageType::${UPPER}\b" src/pipeline/config.cpp \
        && pass "kStageRegistry entry in src/pipeline/config.cpp" \
        || fail "no kStageRegistry entry for StageType::$UPPER in src/pipeline/config.cpp"
fi
grep -q "${CATEGORY}/${NAME}/${NAME}_stage.h" include/fzgpumodules.h \
    && pass "exported from include/fzgpumodules.h" \
    || fail "add #include \"${CATEGORY}/${NAME}/${NAME}_stage.h\" to include/fzgpumodules.h"
if [[ -f "docs/stages/${NAME}.md" ]]; then
    pass "docs/stages/${NAME}.md exists"
    grep -q "stage_${NAME}" "docs/stages/${CATEGORY}.md" 2>/dev/null \
        && pass "listed in docs/stages/${CATEGORY}.md" \
        || fail "add '\\subpage stage_${NAME}' to docs/stages/${CATEGORY}.md"
else
    fail "write docs/stages/${NAME}.md (layout: docs/stages/zigzag.md)"
fi
STAGE_CLASS=$(grep -oP 'class\s+\K[A-Za-z0-9_]+Stage\b' "$HEADER" 2>/dev/null | head -1)
if [[ -n "$STAGE_CLASS" ]] && sed -n '/## \[Unreleased\]/,/## \[[0-9]/p' CHANGELOG.md | grep -q "$STAGE_CLASS\|${STAGE_CLASS%Stage}"; then
    pass "CHANGELOG.md [Unreleased] mentions ${STAGE_CLASS}"
else
    fail "add a CHANGELOG.md [Unreleased] → Added line for ${STAGE_CLASS:-the stage}"
fi
grep -qE 'pipeline_file_round_trip|decompressFromFile' "$TEST_SRC" 2>/dev/null \
    && pass "tests include a fresh-decoder file round trip" \
    || fail "add a file round-trip test (fz_test::pipeline_file_round_trip) to $TEST_SRC"
grep -qE 'saveConfig|loadConfig' "$TEST_SRC" 2>/dev/null \
    && pass "tests include a TOML save/load round trip" \
    || fail "add a TOML saveConfig/loadConfig round-trip test to $TEST_SRC"
if grep -qiE 'prior work|adapted from|direct port|upstream' "$SOURCE" "$HEADER" 2>/dev/null; then
    grep -q '## Acknowledgements' "docs/stages/${NAME}.md" 2>/dev/null \
        && pass "attribution: Acknowledgements section present" \
        || fail "source mentions prior work but docs/stages/${NAME}.md has no ## Acknowledgements"
fi

if [[ "$STATIC" == 1 ]]; then
    echo
    if [[ ${#FAILS[@]} -eq 0 ]]; then echo "RESULT: STATIC PASS (build, tests and sanitizer not run)"; exit 0; fi
    echo "RESULT: FAIL (${#FAILS[@]} problem(s))"; for f in "${FAILS[@]}"; do echo "  - $f"; done; exit 1
fi

# ── Build ─────────────────────────────────────────────────────────────────────
section "Build"
if [[ ! -f third_party/googletest/CMakeLists.txt ]]; then
    git submodule update --init --recursive >/dev/null 2>&1 || true
fi
if [[ ! -f "$BUILD/CMakeCache.txt" ]] || ! grep -q 'BUILD_TESTING:BOOL=ON' "$BUILD/CMakeCache.txt"; then
    echo "  configuring $BUILD (release, tests on, native CUDA arch)"
    cmake --preset release -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=native > /tmp/fz_verify_cfg_$$.log 2>&1 \
        || { fail "cmake configure (log: /tmp/fz_verify_cfg_$$.log)"; }
fi
TARGETS=("$TEST_BIN_NAME" test_stage_registry test_config test_file_io)
if cmake --build "$BUILD" -j"$JOBS" --target "${TARGETS[@]}" > /tmp/fz_verify_build_$$.log 2>&1; then
    pass "built ${TARGETS[*]}"; rm -f /tmp/fz_verify_build_$$.log
else
    fail "build (log: /tmp/fz_verify_build_$$.log)"; grep -E 'error|Error' /tmp/fz_verify_build_$$.log | head -15 | sed 's/^/        /'
fi

# ── Tests ─────────────────────────────────────────────────────────────────────
section "Tests"
if ctest --test-dir "$BUILD" -N -R "^${TEST_BIN_NAME}\$" 2>/dev/null | grep -q 'Total Tests: [1-9]'; then
    if "${WRAP[@]}" ctest --test-dir "$BUILD" --output-on-failure \
            -R "^(${TEST_BIN_NAME}|test_stage_registry|test_config|test_file_io)\$" > /tmp/fz_verify_ctest_$$.log 2>&1; then
        pass "$(grep -E 'tests passed' /tmp/fz_verify_ctest_$$.log | tail -1)"
    else
        fail "ctest (log: /tmp/fz_verify_ctest_$$.log)"; grep -E 'Failed|FAILED|\*\*\*' /tmp/fz_verify_ctest_$$.log | head -15 | sed 's/^/        /'
    fi
else
    fail "ctest has no test named $TEST_BIN_NAME (check tests/stages/CMakeLists.txt and that BUILD_TESTING=ON)"
fi

# ── Compute Sanitizer on the stage's own tests ────────────────────────────────
section "Compute Sanitizer"
BIN=$(find "$BUILD" -type f -name "$TEST_BIN_NAME" -perm -u+x | head -1); [[ -z "$BIN" ]] || BIN=$(realpath "$BIN")
if [[ "${FZ_SKIP_SANITIZER:-0}" == 1 ]]; then
    warn "skipped (FZ_SKIP_SANITIZER=1)"
elif ! command -v compute-sanitizer >/dev/null; then
    warn "compute-sanitizer not on PATH — skipped (run scripts/run_sanitizers.sh where it is)"
elif [[ -z "$BIN" ]]; then
    fail "test binary $TEST_BIN_NAME not found under $BUILD"
else
    for tool in memcheck racecheck initcheck; do
        log="/tmp/fz_verify_${tool}_$$.log"
        if (cd "$(dirname "$BIN")" && "${WRAP[@]}" env CUDA_MODULE_LOADING=EAGER timeout 600 \
                compute-sanitizer --tool "$tool" --error-exitcode=1 --require-cuda-init=no \
                "$BIN" --gtest_fail_if_no_test_linked) > "$log" 2>&1; then
            pass "$tool: clean, $(grep -oE '\[  PASSED  \] [0-9]+ tests?' "$log" | tail -1 | sed 's/.*] //')"; rm -f "$log"
        else
            fail "$tool (log: $log)"; grep -E 'ERROR SUMMARY|RACECHECK SUMMARY|Invalid|Uninitialized|hazard|FAILED' "$log" | head -8 | sed 's/^/        /'
        fi
    done
fi

# ── Result ────────────────────────────────────────────────────────────────────
echo
if [[ ${#FAILS[@]} -eq 0 ]]; then
    echo "RESULT: PASS — now run the full suite once: ctest --preset default"
    exit 0
fi
echo "RESULT: FAIL (${#FAILS[@]} problem(s))"
for f in "${FAILS[@]}"; do echo "  - $f"; done
exit 1
