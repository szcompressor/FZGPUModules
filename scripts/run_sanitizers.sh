#!/usr/bin/env bash
# scripts/run_sanitizers.sh
#
# Run the FZGPUModules test suite under sanitizers.
#
# Modes:
#   compute  — CUDA Compute Sanitizer (memcheck / initcheck / racecheck / synccheck)
#   asan     — Host AddressSanitizer + UndefinedBehaviorSanitizer
#   all      — both compute and asan (default)
#
# Usage examples:
#   ./scripts/run_sanitizers.sh                          # full matrix, build if needed
#   ./scripts/run_sanitizers.sh --mode compute           # compute only
#   ./scripts/run_sanitizers.sh --mode asan              # asan only
#   ./scripts/run_sanitizers.sh --tool memcheck          # one compute tool
#   ./scripts/run_sanitizers.sh --binary test_rle        # one binary only
#   ./scripts/run_sanitizers.sh --filter "RLE*"          # gtest filter
#   ./scripts/run_sanitizers.sh --build                  # force rebuild first
#   ./scripts/run_sanitizers.sh --mode compute --build --tool initcheck --binary test_rze_stage

set -euo pipefail

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_COMPUTE="$ROOT/build-compute"
BUILD_ASAN="$ROOT/build-asan"
JOBS="$(nproc)"

# ── colours ───────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; RESET=''
fi

# ── defaults ──────────────────────────────────────────────────────────────────
MODE="all"
COMPUTE_TOOLS=("memcheck" "initcheck" "racecheck" "synccheck")
SELECTED_TOOL=""
FORCE_BUILD=0
GTEST_FILTER=""
ONLY_BINARY=""

# ── argument parsing ──────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --mode compute|asan|all    Sanitizer mode to run (default: all)
  --tool TOOL                Compute-sanitizer tool: memcheck|initcheck|racecheck|synccheck
                             (default: all four; ignored for asan mode)
  --binary NAME              Run only this test binary (e.g. test_rle)
  --filter PATTERN           Pass --gtest_filter=PATTERN to every test binary
  --build                    Force reconfigure + rebuild before running
  --jobs N                   Parallel jobs for cmake build (default: nproc = $JOBS)
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)    MODE="$2";         shift 2 ;;
        --tool)    SELECTED_TOOL="$2"; shift 2 ;;
        --binary)  ONLY_BINARY="$2";  shift 2 ;;
        --filter)  GTEST_FILTER="$2"; shift 2 ;;
        --build)   FORCE_BUILD=1;     shift   ;;
        --jobs)    JOBS="$2";         shift 2 ;;
        -h|--help) usage; exit 0             ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ -n "$SELECTED_TOOL" ]]; then
    COMPUTE_TOOLS=("$SELECTED_TOOL")
fi

# ── helpers ───────────────────────────────────────────────────────────────────
log()    { echo -e "${BOLD}[sanitizers]${RESET} $*"; }
ok()     { echo -e "  ${GREEN}PASS${RESET}  $*"; }
fail()   { echo -e "  ${RED}FAIL${RESET}  $*"; }
warn()   { echo -e "  ${YELLOW}WARN${RESET}  $*"; }
header() { echo -e "\n${CYAN}${BOLD}── $* ──────────────────────────────────────────${RESET}"; }

# Collect test binaries from a build directory (flat layout under tests/)
collect_binaries() {
    local build_dir="$1"
    local bins=()
    while IFS= read -r -d '' f; do
        bins+=("$f")
    done < <(find "$build_dir/tests" -maxdepth 1 -type f -executable -name 'test_*' -print0 | sort -z)
    echo "${bins[@]}"
}

# ── build helpers ─────────────────────────────────────────────────────────────
build_compute() {
    header "Building compute-sanitizer configuration"
    cmake -S "$ROOT" -B "$BUILD_COMPUTE" \
        -DUSE_SANITIZER=Compute \
        -DCOMPUTE_SANITIZER_DEVICE_DEBUG=OFF \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCUDA_MODULE_LOADING=EAGER \
        2>&1 | grep -E "^--|error:|warning:" || true
    cmake --build "$BUILD_COMPUTE" -j"$JOBS"
}

build_asan() {
    header "Building ASan+UBSan configuration"
    cmake -S "$ROOT" -B "$BUILD_ASAN" \
        -DUSE_SANITIZER=ASanUbsan \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCUDA_MODULE_LOADING=EAGER \
        2>&1 | grep -E "^--|error:|warning:" || true
    cmake --build "$BUILD_ASAN" -j"$JOBS"
}

need_build() {
    local build_dir="$1"
    [[ "$FORCE_BUILD" -eq 1 ]] && return 0
    [[ ! -d "$build_dir" ]] && return 0
    # No test binaries present → need build
    local count
    count=$(find "$build_dir/tests" -maxdepth 1 -type f -executable -name 'test_*' 2>/dev/null | wc -l)
    [[ "$count" -eq 0 ]] && return 0
    return 1
}

# ── compute sanitizer runner ──────────────────────────────────────────────────
run_compute() {
    if need_build "$BUILD_COMPUTE"; then
        build_compute
    else
        log "Using existing compute build at $BUILD_COMPUTE  (pass --build to force rebuild)"
    fi

    local all_pass=1
    # results[tool][binary] = PASS|FAIL
    declare -A results

    for tool in "${COMPUTE_TOOLS[@]}"; do
        header "compute-sanitizer --tool $tool"
        local tool_pass=1

        readarray -d '' binaries < <(find "$BUILD_COMPUTE/tests" -maxdepth 1 \
            -type f -executable -name 'test_*' -print0 | sort -z)

        for bin in "${binaries[@]}"; do
            local name; name="$(basename "$bin")"
            [[ -n "$ONLY_BINARY" && "$name" != "$ONLY_BINARY" ]] && continue

            local cmd=(compute-sanitizer --tool "$tool" --error-exitcode=1)
            [[ "$tool" == "memcheck" ]] && cmd+=(--show-backtrace=yes)
            cmd+=("$bin")
            [[ -n "$GTEST_FILTER" ]] && cmd+=("--gtest_filter=$GTEST_FILTER")

            local out; out=$("${cmd[@]}" 2>&1) && rc=0 || rc=$?
            if [[ $rc -eq 0 ]]; then
                ok "$tool / $name"
                results["$tool/$name"]="PASS"
            else
                fail "$tool / $name"
                results["$tool/$name"]="FAIL"
                tool_pass=0
                all_pass=0
                # Print first ERROR SUMMARY line for context
                echo "$out" | grep -m5 "ERROR SUMMARY\|error\|warning" | sed 's/^/    /' || true
            fi
        done

        if [[ $tool_pass -eq 1 ]]; then
            log "${GREEN}$tool: all PASS${RESET}"
        else
            log "${RED}$tool: FAILURES detected${RESET}"
        fi
    done

    return $(( 1 - all_pass ))
}

# ── asan runner ───────────────────────────────────────────────────────────────
run_asan() {
    if need_build "$BUILD_ASAN"; then
        build_asan
    else
        log "Using existing asan build at $BUILD_ASAN  (pass --build to force rebuild)"
    fi

    header "ASan + UBSan"

    # Locate the ASan shared library
    local libasan; libasan=$(gcc --print-file-name=libasan.so 2>/dev/null || true)
    if [[ -z "$libasan" || ! -f "$libasan" ]]; then
        warn "Could not locate libasan.so via gcc --print-file-name; trying ldconfig"
        libasan=$(ldconfig -p 2>/dev/null | awk '/libasan\.so\./{print $NF; exit}')
    fi
    if [[ -z "$libasan" || ! -f "$libasan" ]]; then
        echo -e "${RED}ERROR: libasan.so not found. Install gcc with ASan support.${RESET}"
        return 1
    fi
    log "Using libasan: $libasan"

    local extra_args=()
    [[ -n "$GTEST_FILTER" ]] && extra_args+=(--gtest-filter "$GTEST_FILTER")
    [[ -n "$ONLY_BINARY"  ]] && extra_args+=(-R "$ONLY_BINARY")

    local rc=0
    LD_PRELOAD="$libasan" \
    ASAN_OPTIONS="detect_leaks=0:abort_on_error=0:protect_shadow_gap=0" \
    UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=0" \
        ctest --test-dir "$BUILD_ASAN" --output-on-failure -j1 "${extra_args[@]}" \
        && rc=0 || rc=$?

    if [[ $rc -eq 0 ]]; then
        log "${GREEN}ASan+UBSan: all PASS${RESET}"
    else
        log "${RED}ASan+UBSan: FAILURES detected${RESET}"
    fi
    return $rc
}

# ── summary ───────────────────────────────────────────────────────────────────
COMPUTE_RC=0
ASAN_RC=0

START=$(date +%s)

case "$MODE" in
    compute) run_compute || COMPUTE_RC=$? ;;
    asan)    run_asan    || ASAN_RC=$?    ;;
    all)     run_compute || COMPUTE_RC=$?
             run_asan    || ASAN_RC=$?    ;;
    *)       echo "Unknown mode: $MODE"; usage; exit 1 ;;
esac

END=$(date +%s)
ELAPSED=$(( END - START ))

header "Summary"
if [[ "$MODE" == "compute" || "$MODE" == "all" ]]; then
    if [[ $COMPUTE_RC -eq 0 ]]; then
        echo -e "  ${GREEN}PASS${RESET}  CUDA Compute Sanitizer (${COMPUTE_TOOLS[*]})"
    else
        echo -e "  ${RED}FAIL${RESET}  CUDA Compute Sanitizer (${COMPUTE_TOOLS[*]})"
    fi
fi
if [[ "$MODE" == "asan" || "$MODE" == "all" ]]; then
    if [[ $ASAN_RC -eq 0 ]]; then
        echo -e "  ${GREEN}PASS${RESET}  ASan + UBSan"
    else
        echo -e "  ${RED}FAIL${RESET}  ASan + UBSan"
    fi
fi
echo ""
log "Total time: ${ELAPSED}s"

OVERALL_RC=$(( COMPUTE_RC | ASAN_RC ))
exit $OVERALL_RC
