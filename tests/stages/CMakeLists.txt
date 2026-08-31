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
#   3. Link the minimal set of libraries needed (prefer fzgmod_modules or
#      fzgmod_modules + fzgmod_mem over the full fzgmod to keep link times
#      short and the test scope honest).
###############################################################################

# Tests include "helpers/fz_test_utils.h" relative to tests/ root.
set(FZ_TEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# ─── Difference encoder ───────────────────────────────────────────────────────
fz_add_test(
    NAME    test_difference
    SOURCES test_difference.cpp
    LIBS    fzgmod_modules fzgmod_mem
    LABELS  stages gpu
)

# ─── RLE encoder ──────────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_rle
    SOURCES test_rle.cpp
    LIBS    fzgmod_modules fzgmod_mem
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
    LIBS    fzgmod_modules fzgmod_mem
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
    LIBS    fzgmod_modules fzgmod_mem
    LABELS  stages gpu
)

# ─── BitshuffleStage GPU round-trip ──────────────────────────────────────────
fz_add_test(
    NAME    test_bitshuffle_stage
    SOURCES test_bitshuffle_stage.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── RZEStage GPU round-trip ─────────────────────────────────────────────────
fz_add_test(
    NAME    test_rze_stage
    SOURCES test_rze_stage.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── RREStage GPU round-trip ─────────────────────────────────────────────────
fz_add_test(
    NAME    test_rre_stage
    SOURCES test_rre_stage.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── RAREStage GPU round-trip ────────────────────────────────────────────────
fz_add_test(
    NAME    test_rare_stage
    SOURCES test_rare_stage.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── RAZEStage GPU round-trip ────────────────────────────────────────────────
fz_add_test(
    NAME    test_raze_stage
    SOURCES test_raze_stage.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── CLOGStage GPU round-trip ────────────────────────────────────────────────
fz_add_test(
    NAME    test_clog_stage
    SOURCES test_clog_stage.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── HCLOGStage GPU round-trip ───────────────────────────────────────────────
fz_add_test(
    NAME    test_hclog_stage
    SOURCES test_hclog_stage.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── TUPLStage GPU round-trip ────────────────────────────────────────────────
fz_add_test(
    NAME    test_tupl_stage
    SOURCES test_tupl_stage.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── LogTransformStage (log-space point-wise relative transform) ─────────────
fz_add_test(
    NAME    test_log_transform
    SOURCES test_log_transform.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── Cdf97Stage (CDF 9/7 wavelet transform, SPERR's DWT) ─────────────────────
fz_add_test(
    NAME    test_cdf97
    SOURCES test_cdf97.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── Speck2DStage (GPU-parallel wavefront SPECK-like coder) ─────────────────
fz_add_test(
    NAME    test_speck2d
    SOURCES test_speck2d.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── MergeStage (structural concat/split) ────────────────────────────────────
fz_add_test(
    NAME    test_merge_stage
    SOURCES test_merge_stage.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── ROIBinSplitStage (structural ROI/background split) ──────────────────────
fz_add_test(
    NAME    test_roibin_split
    SOURCES test_roibin_split.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── Bitpack transform ────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_bitpack
    SOURCES test_bitpack.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── Adaptive bit-plane coder (cuSZp plain mode) ──────────────────────────────
fz_add_test(
    NAME    test_adaptive_bitpack
    SOURCES test_adaptive_bitpack.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── Lorenzo predictor ────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_lorenzo
    SOURCES test_lorenzo.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── Per-tile adaptive multi-order Lorenzo (FSZ prediction stage) ────────────
fz_add_test(
    NAME    test_adaptive_lorenzo
    SOURCES test_adaptive_lorenzo.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── Tiled (dimension-aware) Lorenzo predictor (cuSZp3 delta) ─────────────────
fz_add_test(
    NAME    test_tiled_lorenzo
    SOURCES test_tiled_lorenzo.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── Module-card coverage (index / field consistency check) ──────────────────
# Verifies docs/cards/index.json is in sync with card files on disk and each
# card contains the required top-level fields.  Heavy JSON-schema validation
# is done by validate-cards.py (Z-Hub repo); this C++ test is a fast CI guard.
# Cards moved from modules/*/card.json to docs/cards/fzgm-*.json in v1.0.
fz_add_test(
    NAME    test_module_cards
    SOURCES test_module_cards.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)
target_compile_definitions(test_module_cards PRIVATE
    FZ_CARDS_DIR="${CMAKE_SOURCE_DIR}/docs/cards")

# ─── cuSZp-family pipeline block/tile size sweeps ─────────────────────────────
fz_add_test(
    NAME    test_cuszp_block_sizes
    SOURCES test_cuszp_block_sizes.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── Optional large-data stress tests (env-gated; FZ_LARGE_DATA_TEST=1) ───────
fz_add_test(
    NAME    test_large_data
    SOURCES test_large_data.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu large
)

# ─── Quantizer predictor ─────────────────────────────────────────────────────
fz_add_test(
    NAME    test_quantizer
    SOURCES test_quantizer.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── Type-matrix tests (TYPED_TEST_SUITE) ─────────────────────────────────────
# Parametric round-trip + serialization tests for all <TInput,TCode> pairs
# supported by LorenzoQuantStage and QuantizerStage.  Uses the stage_harness.h
# helpers so each type pair needs only 3-5 lines of test body.
fz_add_test(
    NAME    test_type_matrix
    SOURCES test_type_matrix.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── Huffman stage ────────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_huffman
    SOURCES test_huffman.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)


# ─── ANS stage ────────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_ans
    SOURCES test_ans.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── ADM stage ────────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_adm
    SOURCES test_adm.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── GInterp stage (cuSZ-Hi spline predictor) ─────────────────────────────────
fz_add_test(
    NAME    test_ginterp
    SOURCES test_ginterp.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── BitplaneRZE stage (FZ-GPU fused bitplane transpose + zero-group RZE) ──────
fz_add_test(
    NAME    test_bitplane_rze
    SOURCES test_bitplane_rze.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── GPULZ stage ────────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_gpulz
    SOURCES test_gpulz.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── SZx stage (ultrafast EB, constant-block classification) ──────────────────
fz_add_test(
    NAME    test_szx
    SOURCES test_szx.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)

# ─── SZp / fZ-light stage (quantize + 1-D delta + fixed-length bitpack) ────────
fz_add_test(
    NAME    test_szp
    SOURCES test_szp.cpp
    LIBS    fzgmod_modules fzgmod_mem fzgmod
    LABELS  stages gpu
)
