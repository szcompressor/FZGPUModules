# examples/CMakeLists.txt
# Included unconditionally from the root CMakeLists.txt.
# Each block is guarded by its own option so nothing is built unless requested.

if(BUILD_EXAMPLES)
    # ── Standalone Lorenzo code-distribution analyser ─────────────────────────
    add_executable(analyze_lorenzo analyze_lorenzo.cpp)
    target_link_libraries(analyze_lorenzo PRIVATE fzgmod)
    set_target_properties(analyze_lorenzo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    # ── PREALLOCATE vs MINIMAL memory strategy comparison ─────────────────────
    add_executable(pfpl_example pfpl_pipeline.cpp)
    target_link_libraries(pfpl_example PRIVATE fzgmod)
    set_target_properties(pfpl_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    # ── MINIMAL / PREALLOCATE / CUDA Graph capture three-way comparison ───────
    add_executable(pfpl_graph_capture pfpl_graph_capture.cpp)
    target_link_libraries(pfpl_graph_capture PRIVATE fzgmod)
    set_target_properties(pfpl_graph_capture PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    # ── Simple API: Lorenzo dual-branch pipeline on a real input file ─────────
    add_executable(simple_api_lorenzo_dual_branch simple_api_lorenzo_dual_branch.cpp)
    target_link_libraries(simple_api_lorenzo_dual_branch PRIVATE fzgmod)
    set_target_properties(simple_api_lorenzo_dual_branch PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    # ── Manual stage execution vs Pipeline DAG throughput comparison ──────────
    add_executable(pfpl_manual_vs_dag manual_pipeline.cpp)
    target_link_libraries(pfpl_manual_vs_dag PRIVATE fzgmod)
    set_target_properties(pfpl_manual_vs_dag PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    # Wire NVTX includes into example targets that have profiling annotations
    if(BUILD_PROFILING AND NVTX3_INCLUDE_DIR)
        target_include_directories(pfpl_example PRIVATE ${NVTX3_INCLUDE_DIR})
        target_include_directories(pfpl_graph_capture PRIVATE ${NVTX3_INCLUDE_DIR})
    endif()
endif()

if(BUILD_PROFILING)
    # ── Profiling executable ──────────────────────────────────────────────────
    # Build:  cmake --build build --target fzgmod-profile
    # Run:    nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
    #              -o cldhgh_profile ./build/bin/fzgmod-profile
    #         nsys-ui cldhgh_profile.nsys-rep
    add_executable(fzgmod-profile basic_profiling_example.cpp)
    target_link_libraries(fzgmod-profile PRIVATE fzgmod)
    set_target_properties(fzgmod-profile PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    if(NVTX3_INCLUDE_DIR)
        target_include_directories(fzgmod-profile PRIVATE ${NVTX3_INCLUDE_DIR})
    endif()

    # ── Repeated-compression benchmark ───────────────────────────────────────
    # Measures the throughput benefit of reusing a PREALLOCATE pipeline vs.
    # constructing a fresh MINIMAL pipeline for every compress() call.
    #
    # Build:  cmake --build build --target fzgmod-profile-repeat
    # Run:    ./build/bin/fzgmod-profile-repeat
    #   or:   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
    #              -o repeat_profile ./build/bin/fzgmod-profile-repeat
    #         nsys-ui repeat_profile.nsys-rep
    add_executable(fzgmod-profile-repeat profile_repeat.cpp)
    target_link_libraries(fzgmod-profile-repeat PRIVATE fzgmod)
    set_target_properties(fzgmod-profile-repeat PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    if(NVTX3_INCLUDE_DIR)
        target_include_directories(fzgmod-profile-repeat PRIVATE ${NVTX3_INCLUDE_DIR})
    endif()
endif()
