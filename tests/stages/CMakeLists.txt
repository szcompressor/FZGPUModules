###############################################################################
# tests/stages/CMakeLists.txt
#
# Unit tests for individual compression stages.
# Each test exercises one stage in isolation: forward/inverse correctness,
# all supported data types, configuration modes, header serialization, etc.
#
# All tests are labelled "stages gpu" for CTest label filtering:
#   ctest --preset stages                     # preset-based
#   ctest --test-dir build -L stages          # manual
#
# Adding a new stage test:
#   1. Create test_<stage_name>.cpp in this directory.
#   2. Add an fz_add_test() entry below with LABELS stages gpu.
#   3. Link the minimal set of libraries needed (prefer fzgmod_encoders or
#      fzgmod_predictors + fzgmod_mem over the full fzgmod to keep link times
#      short and the test scope honest).
###############################################################################

# Tests include "helpers/fz_test_utils.h" relative to tests/ root.
set(FZ_TEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# ─── Difference encoder ───────────────────────────────────────────────────────
fz_add_test(
    NAME    test_difference
    SOURCES test_difference.cpp
    LIBS    fzgmod_encoders fzgmod_mem
    LABELS  stages gpu
)

# ─── RLE encoder ──────────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_rle
    SOURCES test_rle.cpp
    LIBS    fzgmod_encoders fzgmod_mem
    LABELS  stages gpu
)

# ─── Zigzag transform (host-side utilities) ───────────────────────────────────
fz_add_test(
    NAME    test_zigzag
    SOURCES test_zigzag.cpp
    LIBS    fzgmod_compile_settings
    LABELS  stages
)

# ─── ZigzagStage GPU round-trip ───────────────────────────────────────────────
fz_add_test(
    NAME    test_zigzag_stage
    SOURCES test_zigzag_stage.cpp
    LIBS    fzgmod_encoders fzgmod_mem
    LABELS  stages gpu
)

# ─── Negabinary transform (host-side utilities) ───────────────────────────────
fz_add_test(
    NAME    test_negabinary
    SOURCES test_negabinary.cpp
    LIBS    fzgmod_compile_settings
    LABELS  stages
)

# ─── NegabinaryStage GPU round-trip ──────────────────────────────────────────
fz_add_test(
    NAME    test_negabinary_stage
    SOURCES test_negabinary_stage.cpp
    LIBS    fzgmod_encoders fzgmod_mem
    LABELS  stages gpu
)

# ─── BitshuffleStage GPU round-trip ──────────────────────────────────────────
fz_add_test(
    NAME    test_bitshuffle_stage
    SOURCES test_bitshuffle_stage.cpp
    LIBS    fzgmod_encoders fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── RZEStage GPU round-trip ─────────────────────────────────────────────────
fz_add_test(
    NAME    test_rze_stage
    SOURCES test_rze_stage.cpp
    LIBS    fzgmod_encoders fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── Bitpack transform ────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_bitpack
    SOURCES test_bitpack.cpp
    LIBS    fzgmod_encoders fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── Lorenzo predictor ────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_lorenzo
    SOURCES test_lorenzo.cpp
    LIBS    fzgmod_predictors fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── Quantizer predictor ─────────────────────────────────────────────────────
fz_add_test(
    NAME    test_quantizer
    SOURCES test_quantizer.cpp
    LIBS    fzgmod_predictors fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── Type-matrix tests (TYPED_TEST_SUITE) ─────────────────────────────────────
# Parametric round-trip + serialization tests for all <TInput,TCode> pairs
# supported by LorenzoStage and QuantizerStage.  Uses the stage_harness.h
# helpers so each type pair needs only 3-5 lines of test body.
fz_add_test(
    NAME    test_type_matrix
    SOURCES test_type_matrix.cpp
    LIBS    fzgmod_predictors fzgmod_mem fzgmod
    LABELS  stages gpu
)
