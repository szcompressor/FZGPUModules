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
#   JOBS=N                build parallelism (default: nproc, capped at 16)
#
# Exit 0 and "RESULT: PASS" only if every check passes. Logs of failed steps are
# kept in a temporary directory that is printed; it is removed on a pass.

set -uo pipefail

STATIC=0
if [[ "${1:-}" == --static ]]; then STATIC=1; shift; fi
NAME="${1:-}"
if [[ -z "$NAME" || "$NAME" == -h || "$NAME" == --help ]]; then
    sed -n '2,15p' "$0"; exit 2
fi
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT" || exit 2
BUILD="${2:-build/release}"
NPROC=$(nproc)
JOBS="${JOBS:-$(( NPROC > 16 ? 16 : NPROC ))}"
read -r -a WRAP <<< "${FZ_GPU_WRAPPER:-}"
LOGS=$(mktemp -d "${TMPDIR:-/tmp}/fz_verify_${NAME}.XXXXXX")

FAILS=()
pass() { printf '  PASS  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; FAILS+=("$*"); }
warn() { printf '  WARN  %s\n' "$*"; }
section() { printf '\n== %s\n' "$*"; }
show() { grep -E "$2" "$1" | head -"${3:-10}" | sed 's/^/        /'; }

finish() {
    echo
    if [[ ${#FAILS[@]} -eq 0 ]]; then
        rm -rf "$LOGS"
        echo "$1"
        exit 0
    fi
    if [[ "$STATIC" == 1 ]]; then   # static failures are fully shown above
        rm -rf "$LOGS"; echo "RESULT: FAIL (${#FAILS[@]} problem(s))"
    else
        echo "RESULT: FAIL (${#FAILS[@]} problem(s)); logs in $LOGS"
    fi
    for f in "${FAILS[@]}"; do echo "  - $f"; done
    exit 1
}

# ── Locate the stage ──────────────────────────────────────────────────────────
MODULE_DIR=$(find modules -mindepth 2 -maxdepth 2 -type d -name "$NAME" | head -1)
if [[ -z "$MODULE_DIR" ]]; then
    echo "No module directory modules/<category>/$NAME — run scripts/new_stage.sh first."
    rm -rf "$LOGS"; exit 2
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
if python3 scripts/check_docs.py > "$LOGS/docs.log" 2>&1; then
    pass "scripts/check_docs.py (includes check_stage_integration.py)"
else
    fail "scripts/check_docs.py"; show "$LOGS/docs.log" 'FAIL|ERROR|error|problem' 20
fi

for f in "$HEADER" "$SOURCE" "$TEST_SRC" "$MODULE_DIR/CMakeLists.txt"; do
    [[ -f "$f" ]] || { fail "missing $f"; continue; }
    n=$(grep -c 'TODO' "$f")
    if [[ "$n" -gt 0 ]]; then fail "$f has $n TODO(s) left from the scaffold"; else pass "no TODOs in $f"; fi
done

if [[ -z "$UPPER" ]]; then
    fail "could not find the StageType used in $SOURCE"
else
    grep -qE "^\s*${UPPER}\s*=.*TODO" include/fzm_format.h \
        && fail "include/fzm_format.h: replace the scaffold's TODO comment on ${UPPER}" \
        || pass "StageType::$UPPER described in include/fzm_format.h"
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
    fail "write docs/stages/${NAME}.md (layout: docs/stages/diff.md)"
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
for table in docs/mainpage.md docs/config_file.md; do
    if [[ -n "$STAGE_CLASS" ]] && grep -q "\b${STAGE_CLASS}\b" "$table"; then
        pass "listed in $table"
    else
        fail "add a ${STAGE_CLASS:-stage} row to the stage table in $table"
    fi
done
ATTR_HITS=$(grep -niE 'prior work|adapted from|direct port|Upstream:|https?://' "$SOURCE" "$HEADER" 2>/dev/null)
if [[ -n "$ATTR_HITS" ]]; then
    if grep -q '## Acknowledgements' "docs/stages/${NAME}.md" 2>/dev/null; then
        pass "attribution: Acknowledgements section present"
    else
        fail "source looks based on prior work, but docs/stages/${NAME}.md has no ## Acknowledgements (Step 8b). Matched:"
        printf '%s\n' "$ATTR_HITS" | head -5 | sed 's/^/        /'
    fi
fi

[[ "$STATIC" == 1 ]] && finish "RESULT: STATIC PASS (build, tests and sanitizer not run)"

# ── Build ─────────────────────────────────────────────────────────────────────
section "Build (-j$JOBS)"
if [[ ! -f third_party/googletest/CMakeLists.txt ]]; then
    git submodule update --init --recursive >/dev/null 2>&1 || true
fi
if [[ ! -f "$BUILD/CMakeCache.txt" ]] || ! grep -q 'BUILD_TESTING:BOOL=ON' "$BUILD/CMakeCache.txt"; then
    echo "  configuring $BUILD (release, tests on, native CUDA arch)"
    cmake --preset release -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=native > "$LOGS/configure.log" 2>&1 \
        || fail "cmake configure (log: $LOGS/configure.log)"
fi
TARGETS=("$TEST_BIN_NAME" test_stage_registry test_config test_file_io)
if cmake --build "$BUILD" -j"$JOBS" --target "${TARGETS[@]}" > "$LOGS/build.log" 2>&1; then
    pass "built ${TARGETS[*]}"
else
    fail "build (log: $LOGS/build.log)"; show "$LOGS/build.log" ' error|Error ' 15
fi

# ── Tests ─────────────────────────────────────────────────────────────────────
section "Tests"
BIN=$(find "$BUILD" -type f -name "$TEST_BIN_NAME" -perm -u+x | head -1); [[ -z "$BIN" ]] || BIN=$(realpath "$BIN")
if ctest --test-dir "$BUILD" -N -R "^${TEST_BIN_NAME}\$" 2>/dev/null | grep -q 'Total Tests: [1-9]'; then
    if "${WRAP[@]}" ctest --test-dir "$BUILD" --output-on-failure \
            -R "^(${TEST_BIN_NAME}|test_stage_registry|test_config|test_file_io)\$" > "$LOGS/ctest.log" 2>&1; then
        pass "$(grep -E 'tests passed' "$LOGS/ctest.log" | tail -1)"
    else
        fail "ctest (log: $LOGS/ctest.log)"; show "$LOGS/ctest.log" 'Failed|FAILED|\*\*\*' 15
    fi
else
    fail "ctest has no test named $TEST_BIN_NAME (check tests/stages/CMakeLists.txt and that BUILD_TESTING=ON)"
fi
if [[ -n "$BIN" ]] && (cd "$(dirname "$BIN")" && "${WRAP[@]}" "$BIN") > "$LOGS/stage_tests.log" 2>&1; then
    n_pass=$(grep -oE '\[  PASSED  \] [0-9]+' "$LOGS/stage_tests.log" | grep -oE '[0-9]+$' | tail -1)
    n_skip=$(grep -oE '\[  SKIPPED \] [0-9]+' "$LOGS/stage_tests.log" | grep -oE '[0-9]+$' | tail -1)
    pass "$TEST_BIN_NAME: ${n_pass:-?} test cases"
    [[ -z "$n_skip" || "$n_skip" == 0 ]] || warn "$TEST_BIN_NAME skipped $n_skip test case(s) — check they are meant to skip here"
fi

# ── Poisoned allocations (only if this build supports FZ_POISON_ALLOCATIONS) ──
section "Poisoned allocations"
if ! grep -rqs 'FZ_POISON_ALLOCATIONS' src include; then
    warn "unavailable in this build (no FZ_POISON_ALLOCATIONS support); buffer-contents rules unchecked"
else
    for byte in 0xA5 0x00; do
        if [[ -n "$BIN" ]] && (cd "$(dirname "$BIN")" && "${WRAP[@]}" env FZ_POISON_ALLOCATIONS=$byte "$BIN") > "$LOGS/poison_$byte.log" 2>&1; then
            pass "FZ_POISON_ALLOCATIONS=$byte"
        else
            fail "FZ_POISON_ALLOCATIONS=$byte — some byte is read before it is written (log: $LOGS/poison_$byte.log)"
            show "$LOGS/poison_$byte.log" 'FAILED|Failure' 8
        fi
    done
fi

# ── Compute Sanitizer on the stage's own tests ────────────────────────────────
section "Compute Sanitizer"
if [[ "${FZ_SKIP_SANITIZER:-0}" == 1 ]]; then
    warn "skipped (FZ_SKIP_SANITIZER=1)"
elif ! command -v compute-sanitizer >/dev/null; then
    warn "compute-sanitizer not on PATH — skipped (run scripts/run_sanitizers.sh where it is)"
elif [[ -z "$BIN" ]]; then
    fail "test binary $TEST_BIN_NAME not found under $BUILD"
else
    for tool in memcheck racecheck initcheck; do
        log="$LOGS/$tool.log"
        if (cd "$(dirname "$BIN")" && "${WRAP[@]}" env CUDA_MODULE_LOADING=EAGER timeout 600 \
                compute-sanitizer --tool "$tool" --error-exitcode=1 --require-cuda-init=no \
                "$BIN" --gtest_fail_if_no_test_linked) > "$log" 2>&1; then
            pass "$tool: clean, $(grep -oE '\[  PASSED  \] [0-9]+ tests?' "$log" | tail -1 | sed 's/.*] //')"
        else
            fail "$tool (log: $log)"; show "$log" 'ERROR SUMMARY|RACECHECK SUMMARY|Invalid|Uninitialized|hazard|FAILED' 8
        fi
    done
fi

finish "RESULT: PASS — now rebuild everything and run the full suite once:
  cmake --build $BUILD -j$JOBS && ctest --preset default"
