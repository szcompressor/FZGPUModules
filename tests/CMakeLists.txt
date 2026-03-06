###############################################################################
# tests/CMakeLists.txt
#
# One executable per test file.  Each target:
#   - links the relevant FZModules library + GTest::gtest_main
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
    target_link_libraries(${ARG_NAME} PRIVATE fzmod_compile_settings)

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
    LIBS    fzmod_encoders fzmod_mem
)

# ─── RLE encoder ──────────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_rle
    SOURCES test_rle.cpp
    LIBS    fzmod_encoders fzmod_mem
)

# ─── Lorenzo predictor ────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_lorenzo
    SOURCES test_lorenzo.cpp
    LIBS    fzmod_predictors fzmod_mem
)

# ─── Full pipeline (compress + decompress) ────────────────────────────────────
fz_add_test(
    NAME    test_pipeline
    SOURCES test_pipeline.cpp
    LIBS    fzmod
)
