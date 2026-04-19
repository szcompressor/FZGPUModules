###############################################################################
# tests/pipeline/CMakeLists.txt
#
# Integration tests for the pipeline layer: DAG execution, memory strategies,
# file I/O, graph capture, buffer liveness, ownership semantics, and
# pipeline-level error handling.
#
# These tests exercise multiple stages wired together through the Pipeline and
# CompressionDAG APIs, and typically depend on the full fzgmod library.
#
# All tests are labelled "pipeline gpu" for CTest label filtering:
#   ctest --preset pipeline                   # preset-based
#   ctest --test-dir build -L pipeline        # manual
###############################################################################

# Tests include "helpers/fz_test_utils.h" relative to tests/ root.
set(FZ_TEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# ─── Full pipeline (compress + decompress) ────────────────────────────────────
fz_add_test(
    NAME    test_pipeline
    SOURCES test_pipeline.cpp
    LIBS    fzgmod
    LABELS  pipeline gpu
)

# ─── Pipeline error-handling / invalid-usage guards ───────────────────────────
fz_add_test(
    NAME    test_pipeline_errors
    SOURCES test_pipeline_errors.cpp
    LIBS    fzgmod
    LABELS  pipeline gpu
)

# ─── Memory strategy comparison (MINIMAL / PREALLOCATE) ──────────────────────
fz_add_test(
    NAME    test_memory_strategies
    SOURCES test_memory_strategies.cpp
    LIBS    fzgmod
    LABELS  pipeline gpu
)

# ─── File I/O + FZM header format ─────────────────────────────────────────────
fz_add_test(
    NAME    test_file_io
    SOURCES test_file_io.cpp
    LIBS    fzgmod
    LABELS  pipeline gpu
)

# ─── CUDA Graph capture correctness and replay ────────────────────────────────
fz_add_test(
    NAME    test_graph_capture
    SOURCES test_graph_capture.cpp
    LIBS    fzgmod
    LABELS  pipeline gpu
)

# ─── Buffer liveness analysis and level structure ─────────────────────────────
fz_add_test(
    NAME    test_liveness
    SOURCES test_liveness.cpp
    LIBS    fzgmod fz_test_stages
    LABELS  pipeline gpu
)

# ─── Buffer coloring (PREALLOCATE memory reduction) ───────────────────────────
fz_add_test(
    NAME    test_buffer_coloring
    SOURCES test_buffer_coloring.cpp
    LIBS    fzgmod
    LABELS  pipeline gpu
)

# ─── Pointer ownership and lifetime semantics ─────────────────────────────────
fz_add_test(
    NAME    test_ownership
    SOURCES test_ownership.cpp
    LIBS    fzgmod
    LABELS  pipeline gpu
)

# ─── Data pattern correctness (corner-case inputs) ────────────────────────────
fz_add_test(
    NAME    test_data_patterns
    SOURCES test_data_patterns.cpp
    LIBS    fzgmod
    LABELS  pipeline gpu
)

# ─── Profiling API (enableProfiling, getLastPerfResult, printDAG) ─────────────
fz_add_test(
    NAME    test_profiling
    SOURCES test_profiling.cpp
    LIBS    fzgmod
    LABELS  pipeline gpu
)

# ─── Golden-file format regression tests ──────────────────────────────────────
# Reads committed .fzm reference files from tests/golden/ and verifies:
#   (1) version field matches current FZM_VERSION
#   (2) decompressFromFile produces output within original error bound
#   (3) re-compression produces a compressed size within ±5% of the reference
# Tests skip gracefully (GTEST_SKIP) if the golden file doesn't exist.
# Regenerate: cmake --build --preset release --target generate_golden_files
fz_add_test(
    NAME    test_golden
    SOURCES test_golden.cpp
    LIBS    fzgmod
    LABELS  pipeline gpu golden
)
# Inject the source-tree path to tests/golden/ as a compile definition so
# test_golden.cpp can find the committed reference files regardless of where
# the build directory lives.
target_compile_definitions(test_golden PRIVATE
    FZGM_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../golden"
)
