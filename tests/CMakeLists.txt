###############################################################################
# tests/CMakeLists.txt
#
# One executable per test file.  Each target:
#   - links the relevant FZGPUModules library + GTest::gtest_main
#   - is registered with CTest via add_test()
#
# Run all tests:
#   cmake --build build --target all
#   ctest --test-dir build --output-on-failure
###############################################################################

# Shared include path for test helpers
set(FZ_TEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/helpers)

# ─── helper function ──────────────────────────────────────────────────────────
# fz_add_test(NAME <name> SOURCES <file...> LIBS <target...>)
#   Creates an executable, links deps, registers with CTest.
function(fz_add_test)
    cmake_parse_arguments(ARG "" "NAME" "SOURCES;LIBS" ${ARGN})

    add_executable(${ARG_NAME} ${ARG_SOURCES})

    target_include_directories(${ARG_NAME} PRIVATE ${FZ_TEST_INCLUDE_DIR})

    target_link_libraries(${ARG_NAME}
        PRIVATE
            ${ARG_LIBS}
            GTest::gtest_main
            CUDA::cudart
    )

    # Reuse the same compile settings (C++17, CUDA arch flags, etc.)
    target_link_libraries(${ARG_NAME} PRIVATE fzgmod_compile_settings)

    add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME})

    # Allow tests to find CUDA device at runtime
    set_tests_properties(${ARG_NAME} PROPERTIES
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
    )
endfunction()

# ─── Difference encoder ───────────────────────────────────────────────────────
fz_add_test(
    NAME    test_difference
    SOURCES test_difference.cpp
    LIBS    fzgmod_encoders fzgmod_mem
)

# ─── RLE encoder ──────────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_rle
    SOURCES test_rle.cpp
    LIBS    fzgmod_encoders fzgmod_mem
)

# ─── Lorenzo predictor ────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_lorenzo
    SOURCES test_lorenzo.cpp
    LIBS    fzgmod_predictors fzgmod_mem fzgmod
)

# ─── Quantizer predictor (ABS / NOA / REL error-bound modes) ─────────────────
fz_add_test(
    NAME    test_quantizer
    SOURCES test_quantizer.cpp
    LIBS    fzgmod_predictors fzgmod_mem
)

# ─── Full pipeline (compress + decompress) ────────────────────────────────────
fz_add_test(
    NAME    test_pipeline
    SOURCES test_pipeline.cpp
    LIBS    fzgmod
)

# ─── Memory strategy comparison (MINIMAL / PIPELINE / PREALLOCATE) ────────────
fz_add_test(
    NAME    test_memory_strategies
    SOURCES test_memory_strategies.cpp
    LIBS    fzgmod
)

# ─── Pipeline error-handling / invalid-usage guards ───────────────────────────
fz_add_test(
    NAME    test_pipeline_errors
    SOURCES test_pipeline_errors.cpp
    LIBS    fzgmod
)

# ─── Data pattern correctness (corner-case inputs) ───────────────────────────
fz_add_test(
    NAME    test_data_patterns
    SOURCES test_data_patterns.cpp
    LIBS    fzgmod
)

# ─── File I/O + FZM header format ─────────────────────────────────────────────
fz_add_test(
    NAME    test_file_io
    SOURCES test_file_io.cpp
    LIBS    fzgmod
)

# ─── Zigzag / TCMS transform utilities (host-side) ───────────────────────────
fz_add_test(
    NAME    test_zigzag
    SOURCES test_zigzag.cpp
    LIBS    fzgmod_compile_settings
)

# ─── Negabinary transform utilities (host-side) ───────────────────────────────
fz_add_test(
    NAME    test_negabinary
    SOURCES test_negabinary.cpp
    LIBS    fzgmod_compile_settings
)

# ─── ZigzagStage GPU round-trip ───────────────────────────────────────────────
fz_add_test(
    NAME    test_zigzag_stage
    SOURCES test_zigzag_stage.cpp
    LIBS    fzgmod_encoders fzgmod_mem
)

# ─── NegabinaryStage GPU round-trip ──────────────────────────────────────────
fz_add_test(
    NAME    test_negabinary_stage
    SOURCES test_negabinary_stage.cpp
    LIBS    fzgmod_encoders fzgmod_mem
)

# ─── BitshuffleStage GPU round-trip ──────────────────────────────────────────
fz_add_test(
    NAME    test_bitshuffle_stage
    SOURCES test_bitshuffle_stage.cpp
    LIBS    fzgmod_encoders fzgmod_mem
)
# ─── RZEStage GPU round-trip ─────────────────────────────────────────────────
fz_add_test(
    NAME    test_rze_stage
    SOURCES test_rze_stage.cpp
    LIBS    fzgmod_encoders fzgmod_mem
)