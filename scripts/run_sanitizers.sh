#!/usr/bin/env bash
# Run registered tests serially under Compute Sanitizer or host ASan+UBSan.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_COMPUTE="$ROOT/build-compute"
BUILD_ASAN="$ROOT/build-asan"
JOBS="$(nproc)"
MODE=all
COMPUTE_TOOLS=(memcheck initcheck racecheck synccheck)
SELECTED_TOOL=""
FORCE_BUILD=0
CASE_FILTER="${GTEST_FILTER:-*}"
ONLY_BINARY=""
TIMEOUT_SECONDS=300

usage() {
    cat <<HELP
Usage: $(basename "$0") [OPTIONS]
  --mode compute|asan|all    Sanitizer mode (default: all)
  --tool TOOL                memcheck|initcheck|racecheck|synccheck
                             (default: all four; ignored for asan mode)
  --binary NAME              Run only this registered test binary
  --filter PATTERN           Google Test filter; an empty selection fails
  --build                    Force reconfigure and rebuild before running
  --jobs N                   Build jobs (default: $JOBS); tests remain serial
  --timeout-seconds N        Per-test/listing timeout (default: 300); kill after 10s
  -h, --help                 Show this help
HELP
}
log() { printf '[sanitizers] %s\n' "$*"; }
error() { printf '[sanitizers] ERROR: %s\n' "$*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode|--tool|--binary|--filter|--jobs|--timeout-seconds)
            [[ $# -ge 2 ]] || { error "$1 requires a value"; exit 1; }
            case "$1" in
                --mode) MODE="$2" ;;
                --tool) SELECTED_TOOL="$2" ;;
                --binary) ONLY_BINARY="$2" ;;
                --filter) CASE_FILTER="$2" ;;
                --jobs) JOBS="$2" ;;
                --timeout-seconds) TIMEOUT_SECONDS="$2" ;;
            esac
            shift 2 ;;
        --build) FORCE_BUILD=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) error "Unknown option: $1"; usage; exit 1 ;;
    esac
done
case "$MODE" in compute|asan|all) ;; *) error "Unknown mode: $MODE"; exit 1 ;; esac
case "$SELECTED_TOOL" in
    "") ;;
    memcheck|initcheck|racecheck|synccheck) COMPUTE_TOOLS=("$SELECTED_TOOL") ;;
    *) error "Unknown compute tool: $SELECTED_TOOL"; exit 1 ;;
esac
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { error '--jobs must be a positive integer'; exit 1; }
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
    error '--timeout-seconds must be a positive integer'; exit 1;
}
for dependency in cmake ctest python3 timeout; do
    command -v "$dependency" >/dev/null || { error "$dependency is required"; exit 1; }
done
timeout --version | grep -q 'GNU coreutils' || { error 'GNU timeout is required'; exit 1; }
if [[ "$MODE" != asan ]]; then
    command -v compute-sanitizer >/dev/null || { error 'compute-sanitizer is required'; exit 1; }
fi

build_config() {
    local build_dir="$1" sanitizer="$2" build_type="$3"
    log "Configuring $sanitizer at $build_dir"
    # Direct compute invocations supply their own wrapper; never wrap ASan tests.
    cmake -S "$ROOT" -B "$build_dir" \
        -DFZGMOD_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_TESTING=ON \
        -DFZGMOD_COMPUTE_SANITIZER_WRAPPER=OFF \
        -DUSE_SANITIZER="$sanitizer" -DCMAKE_BUILD_TYPE="$build_type" \
        -DCOMPUTE_SANITIZER_DEVICE_DEBUG=OFF -DCUDA_MODULE_LOADING=EAGER || return $?
    cmake --build "$build_dir" -j"$JOBS" || return $?
}

need_build() {
    local build_dir="$1" sanitizer="$2"
    [[ "$FORCE_BUILD" -eq 1 ]] && return 0
    local cache="$build_dir/CMakeCache.txt"
    [[ -f "$cache" && -f "$build_dir/CTestTestfile.cmake" ]] || return 0
    # Reject cached configurations that would silently omit tests or double-wrap.
    grep -qx 'BUILD_TESTING:BOOL=ON' "$cache" || return 0
    grep -qEx "USE_SANITIZER:[^=]+=$sanitizer" "$cache" || return 0
    grep -qEx 'FZGMOD_BACKEND:[^=]+=CUDA' "$cache" || return 0
    grep -qx 'FZGMOD_COMPUTE_SANITIZER_WRAPPER:BOOL=OFF' "$cache" || return 0
    return 1
}

# CTest is the source of truth, including binaries in nested directories.
# Store paths in the caller's `binaries` array; reject absent/unbuilt selections.
collect_binaries() {
    local build_dir="$1" json paths
    json=$(ctest --test-dir "$build_dir" --show-only=json-v1) || return $?
    paths=$(python3 -c '
import json, os, sys
try:
    tests = json.load(sys.stdin)["tests"]
    if sys.argv[1]:
        tests = [t for t in tests if t["name"] == sys.argv[1]]
    if not tests:
        raise ValueError("no registered tests match the requested selection")
    for test in tests:
        command = test.get("command", [])
        if len(command) != 1 or not os.access(command[0], os.X_OK):
            raise ValueError("test is missing, unbuilt, or unexpectedly wrapped: " + test["name"])
        properties = {p["name"]: p["value"] for p in test.get("properties", [])}
        working_dir = properties.get("WORKING_DIRECTORY", os.path.dirname(command[0]))
        if not os.path.isdir(working_dir):
            raise ValueError("test working directory is missing: " + test["name"])
        print(command[0])
        print(working_dir)
except (KeyError, ValueError) as error:
    print("[sanitizers] ERROR: " + str(error), file=sys.stderr)
    sys.exit(1)
' "$ONLY_BINARY" <<<"$json") || return $?
    local fields=() index
    mapfile -t fields <<<"$paths"
    for ((index = 0; index < ${#fields[@]}; index += 2)); do
        binaries+=("${fields[index]}")
        working_dirs["${fields[index]}"]="${fields[index + 1]}"
    done
}

# Validate Google Test filtering before a zero-case binary can report success.
# Disabled cases do not count toward the selection.
filter_binaries() {
    [[ -n "$CASE_FILTER" ]] || { error '--filter must not be empty'; return 1; }
    [[ "$CASE_FILTER" != '*' ]] || return 0
    local selected=() bin listing
    for bin in "${binaries[@]}"; do
        listing=$(cd "${working_dirs["$bin"]}" && timeout --kill-after=10s "${TIMEOUT_SECONDS}s" \
            "${test_environment[@]}" "$bin" \
            --gtest_list_tests "--gtest_filter=$CASE_FILTER" 2>&1) || {
            printf '%s\n' "$listing" >&2
            error "Cannot enumerate filtered cases in $bin"
            return 1
        }
        if awk '
            /^[^[:space:]]+\./ { disabled = ($1 ~ /(^|\/)DISABLED_/) }
            /^  [^[:space:]]/ && !disabled && $0 !~ /^  (.*\/)?DISABLED_/ { found = 1 }
            END { exit !found }
        ' <<<"$listing"; then
            selected+=("$bin")
        fi
    done
    [[ ${#selected[@]} -gt 0 ]] || { error "No enabled cases match --filter $CASE_FILTER"; return 1; }
    binaries=("${selected[@]}")
}

run_compute() {
    if need_build "$BUILD_COMPUTE" Compute; then
        build_config "$BUILD_COMPUTE" Compute RelWithDebInfo || return $?
    else
        log "Using existing compute build (pass --build to refresh)"
    fi
    local binaries=() test_environment=(env CUDA_MODULE_LOADING=EAGER "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}")
    local -A working_dirs=()
    collect_binaries "$BUILD_COMPUTE" || return $?
    filter_binaries || return $?
    local tool bin rc status=0 logfile
    local log_dir="$BUILD_COMPUTE/sanitizer-logs/$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mkdir -p "$log_dir" || return $?
    log "Selected ${#binaries[@]} registered compute test binaries; logs: $log_dir"
    for tool in "${COMPUTE_TOOLS[@]}"; do
        for bin in "${binaries[@]}"; do
            local cmd=(timeout --kill-after=10s "${TIMEOUT_SECONDS}s"
                "${test_environment[@]}" compute-sanitizer --tool "$tool"
                --error-exitcode=1 --require-cuda-init=no)
            [[ "$tool" != memcheck ]] || cmd+=(--show-backtrace=yes)
            cmd+=("$bin" --gtest_fail_if_no_test_linked)
            cmd+=("--gtest_filter=$CASE_FILTER")
            logfile="$log_dir/$tool-$(basename "$bin").log"
            printf '%q ' "${cmd[@]}" >"$logfile" || return $?
            printf '\n' >>"$logfile" || return $?
            if (cd "${working_dirs["$bin"]}" && "${cmd[@]}") >>"$logfile" 2>&1; then
                log "PASS $tool / $(basename "$bin")"
            else
                rc=$?
                log "FAIL $tool / $(basename "$bin") (exit $rc)"
                if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
                    printf '[sanitizers] Timed out after %ss (10s kill grace)\n' \
                        "$TIMEOUT_SECONDS" >>"$logfile"
                fi
                cat "$logfile" >&2
                status=1
            fi
        done
    done
    return "$status"
}

run_asan() {
    if need_build "$BUILD_ASAN" ASanUbsan; then
        build_config "$BUILD_ASAN" ASanUbsan Debug || return $?
    else
        log "Using existing ASan build (pass --build to refresh)"
    fi
    local compiler libasan
    compiler=$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "$BUILD_ASAN/CMakeCache.txt")
    [[ -n "$compiler" ]] || { error 'Cannot determine ASan build compiler'; return 1; }
    libasan=$("$compiler" --print-file-name=libasan.so) || return $?
    [[ -f "$libasan" ]] || { error "ASan runtime not found: $libasan"; return 1; }
    local binaries=() test_environment=(env "LD_PRELOAD=$libasan"
        ASAN_OPTIONS=detect_leaks=0:halt_on_error=1:abort_on_error=1:protect_shadow_gap=0
        UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1
        CUDA_MODULE_LOADING=EAGER "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}")
    local -A working_dirs=()
    collect_binaries "$BUILD_ASAN" || return $?
    filter_binaries || return $?
    local bin rc status=0 logfile
    local log_dir="$BUILD_ASAN/sanitizer-logs/$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mkdir -p "$log_dir" || return $?
    log "Selected ${#binaries[@]} registered ASan+UBSan test binaries; logs: $log_dir"
    # Run binaries directly, so the runtime is preloaded only into test processes
    # and filtering cannot accidentally select an unrelated CTest name.
    for bin in "${binaries[@]}"; do
        local cmd=(timeout --kill-after=10s "${TIMEOUT_SECONDS}s"
            "${test_environment[@]}" "$bin" --gtest_fail_if_no_test_linked)
        cmd+=("--gtest_filter=$CASE_FILTER")
        logfile="$log_dir/asan-ubsan-$(basename "$bin").log"
        printf '%q ' "${cmd[@]}" >"$logfile" || return $?
        printf '\n' >>"$logfile" || return $?
        if (cd "${working_dirs["$bin"]}" && "${cmd[@]}") >>"$logfile" 2>&1; then
            log "PASS ASan+UBSan / $(basename "$bin")"
        else
            rc=$?
            log "FAIL ASan+UBSan / $(basename "$bin") (exit $rc)"
            if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
                printf '[sanitizers] Timed out after %ss (10s kill grace)\n' \
                    "$TIMEOUT_SECONDS" >>"$logfile"
            fi
            cat "$logfile" >&2
            status=1
        fi
    done
    return "$status"
}

COMPUTE_RC=0
ASAN_RC=0
START=$(date +%s)
case "$MODE" in
    compute) run_compute || COMPUTE_RC=$? ;;
    asan) run_asan || ASAN_RC=$? ;;
    all) run_compute || COMPUTE_RC=$?; run_asan || ASAN_RC=$? ;;
esac
if [[ "$MODE" != asan ]]; then log "Compute Sanitizer status: $COMPUTE_RC"; fi
if [[ "$MODE" != compute ]]; then log "ASan+UBSan status: $ASAN_RC"; fi
log "Total time: $(($(date +%s) - START))s"
[[ "$COMPUTE_RC" -eq 0 && "$ASAN_RC" -eq 0 ]]
