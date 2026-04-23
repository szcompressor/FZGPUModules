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
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── PREALLOCATE vs MINIMAL memory strategy comparison ─────────────────────────
add_executable(pfpl_example pfpl_pipeline.cpp)
target_link_libraries(pfpl_example PRIVATE fzgmod)
set_target_properties(pfpl_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── MINIMAL / PREALLOCATE / CUDA Graph capture three-way comparison ───────────
add_executable(pfpl_graph_capture pfpl_graph_capture.cpp)
target_link_libraries(pfpl_graph_capture PRIVATE fzgmod)
set_target_properties(pfpl_graph_capture PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── Simple API: Lorenzo dual-branch pipeline on a real input file ─────────────
add_executable(simple_api_lorenzo_dual_branch simple_api_lorenzo_dual_branch.cpp)
target_link_libraries(simple_api_lorenzo_dual_branch PRIVATE fzgmod)
set_target_properties(simple_api_lorenzo_dual_branch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── Manual stage execution vs Pipeline DAG throughput comparison ──────────────
add_executable(pfpl_manual_vs_dag manual_pipeline.cpp)
target_link_libraries(pfpl_manual_vs_dag PRIVATE fzgmod)
set_target_properties(pfpl_manual_vs_dag PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── File IO: compress to .fzm file and decompress from file ───────────────────
add_executable(file_io_example file_io_example.cpp)
target_link_libraries(file_io_example PRIVATE fzgmod)
set_target_properties(file_io_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── Ownership: pool-owned vs caller-owned compress and decompress output ──────
add_executable(ownership_example ownership_example.cpp)
target_link_libraries(ownership_example PRIVATE fzgmod)
set_target_properties(ownership_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── Debugging and logging: Logger, printDAG, profiling, bounds checking ───────
add_executable(debug_logging debug_logging.cpp)
target_link_libraries(debug_logging PRIVATE fzgmod)
set_target_properties(debug_logging PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")
