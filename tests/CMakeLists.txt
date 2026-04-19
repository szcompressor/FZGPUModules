###############################################################################
# tests/CMakeLists.txt
#
# Test suite root.  Tests are grouped by scope into subdirectories:
#
#   stages/    — individual stage unit tests (encoders, predictors, transforms)
#   pipeline/  — pipeline-level integration tests (DAG, memory, file I/O, etc.)
#   test_cli.cpp — top-level CLI smoke test (no subdirectory; depends on full lib)
#
# Quick-start via CMake presets (see CMakePresets.json in repo root):
#
#   cmake --preset release && cmake --build --preset release
#   ctest --preset default           # all tests, release build
#   ctest --preset stages            # stage unit tests only
#   ctest --preset pipeline          # pipeline integration tests only
#   ctest --preset asan              # full suite, ASan + UBSan
#   ctest --preset compute-san       # full suite, compute-sanitizer memcheck
#
# Without presets:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ctest --test-dir build --output-on-failure
#   ctest --test-dir build -L stages --output-on-failure
#   ctest --test-dir build -L pipeline --output-on-failure
###############################################################################

# Tests include "helpers/fz_test_utils.h" relative to this directory.
# Set to CMAKE_CURRENT_SOURCE_DIR (tests/) so the "helpers/" prefix resolves.
set(FZ_TEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
set(FZ_TEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ─── Compute Sanitizer wrapper ────────────────────────────────────────────────
# When ON, every test executable gets CROSSCOMPILING_EMULATOR set to
# "compute-sanitizer --tool memcheck --error-exitcode 1" so that ctest
# automatically wraps each test invocation.  Enabled by the compute-san
# CMakePreset (cacheVariable FZGMOD_COMPUTE_SANITIZER_WRAPPER=ON).
#
# Requires compute-sanitizer to be on PATH at test time.
option(FZGMOD_COMPUTE_SANITIZER_WRAPPER
    "Wrap each test with 'compute-sanitizer --tool memcheck' via CROSSCOMPILING_EMULATOR"
    OFF
)

# ─── helper function ──────────────────────────────────────────────────────────
# fz_add_test(NAME <name> SOURCES <file...> LIBS <target...> LABELS <label...>)
#   Creates an executable, links deps, registers with CTest.
#   LABELS is optional; values are attached to the CTest test so -L / --preset
#   filters work (e.g. LABELS stages gpu, LABELS pipeline gpu).
#   Defined here so subdirectory CMakeLists files can use it.
function(fz_add_test)
    cmake_parse_arguments(ARG "" "NAME" "SOURCES;LIBS;LABELS" ${ARGN})

    add_executable(${ARG_NAME} ${ARG_SOURCES})

    target_include_directories(${ARG_NAME} PRIVATE ${FZ_TEST_INCLUDE_DIR})

    target_link_libraries(${ARG_NAME}
        PRIVATE
            ${ARG_LIBS}
            GTest::gtest_main
            CUDA::cudart
    )

    target_link_libraries(${ARG_NAME} PRIVATE fzgmod_compile_settings)

    # Wrap each test invocation with compute-sanitizer when requested.
    # CTest reads CROSSCOMPILING_EMULATOR from the target and prepends it to the
    # test command in CTestTestfile.cmake — no changes to individual test files needed.
    if(FZGMOD_COMPUTE_SANITIZER_WRAPPER)
        set_target_properties(${ARG_NAME} PROPERTIES
            CROSSCOMPILING_EMULATOR
                "compute-sanitizer;--tool;memcheck;--error-exitcode;1"
        )
    endif()

    add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME})

    set(FZ_TEST_ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
    list(APPEND FZ_TEST_ENVIRONMENT "CUDA_MODULE_LOADING=${FZGMOD_CUDA_MODULE_LOADING_VALUE}")

    set_tests_properties(${ARG_NAME} PROPERTIES
        ENVIRONMENT "${FZ_TEST_ENVIRONMENT}"
        LABELS      "${ARG_LABELS}"
    )
endfunction()

# ─── Test scaffold stages (PassThrough, Scale, Split, Merge) ─────────────────
# Header-only except for ScaleStage, which has a CUDA kernel.
# Build as a small object library so any test or dev example can link it.
add_library(fz_test_stages OBJECT helpers/test_stages.cu)
target_include_directories(fz_test_stages PUBLIC ${FZ_TEST_INCLUDE_DIR})
target_link_libraries(fz_test_stages PUBLIC fzgmod CUDA::cudart)
target_link_libraries(fz_test_stages PRIVATE fzgmod_compile_settings)

# ─── Stage unit tests ─────────────────────────────────────────────────────────
add_subdirectory(stages)

# ─── Pipeline integration tests ───────────────────────────────────────────────
add_subdirectory(pipeline)

# ─── CLI smoke tests ──────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_cli
    SOURCES test_cli.cpp
    LIBS    fzgmod
    LABELS  cli gpu
)
