# examples/CMakeLists.txt
# Included unconditionally from the root CMakeLists.txt.
# Build with: cmake -DBUILD_EXAMPLES=ON ..

if(NOT BUILD_EXAMPLES)
  return()
endif()

# ── Standalone Lorenzo code-distribution analyser ─────────────────────────────
add_executable(analyze_lorenzo analyze_lorenzo.cpp)
target_link_libraries(analyze_lorenzo PRIVATE fzgmod)
set_target_properties(analyze_lorenzo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ── PREALLOCATE vs MINIMAL memory strategy comparison ─────────────────────────
add_executable(pfpl_example pfpl_pipeline.cpp)
target_link_libraries(pfpl_example PRIVATE fzgmod)
set_target_properties(pfpl_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ── MINIMAL / PREALLOCATE / CUDA Graph capture three-way comparison ───────────
add_executable(pfpl_graph_capture pfpl_graph_capture.cpp)
target_link_libraries(pfpl_graph_capture PRIVATE fzgmod)
set_target_properties(pfpl_graph_capture PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ── Simple API: Lorenzo dual-branch pipeline on a real input file ─────────────
add_executable(simple_api_lorenzo_dual_branch simple_api_lorenzo_dual_branch.cpp)
target_link_libraries(simple_api_lorenzo_dual_branch PRIVATE fzgmod)
set_target_properties(simple_api_lorenzo_dual_branch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ── Caller-allocated output API example ──────────────────────────────────────
add_executable(caller_allocated_output caller_allocated_output.cpp)
target_link_libraries(caller_allocated_output PRIVATE fzgmod)
set_target_properties(caller_allocated_output PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ── Manual stage execution vs Pipeline DAG throughput comparison ──────────────
add_executable(pfpl_manual_vs_dag manual_pipeline.cpp)
target_link_libraries(pfpl_manual_vs_dag PRIVATE fzgmod)
set_target_properties(pfpl_manual_vs_dag PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
