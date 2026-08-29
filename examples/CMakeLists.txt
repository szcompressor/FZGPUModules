# examples/CMakeLists.txt
# Included unconditionally from the root CMakeLists.txt.
# Build with: cmake -DBUILD_EXAMPLES=ON ..

if(NOT BUILD_EXAMPLES)
  return()
endif()

# ── Hello-world introduction: LorenzoQuant → Huffman compress+decompress+verify
add_executable(minimal_intro minimal_intro.cpp)
target_link_libraries(minimal_intro PRIVATE fzgmod)
set_target_properties(minimal_intro PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── TOML config: load preset, saveConfig, loadConfig round-trip ──────────────
add_executable(toml_config toml_config.cpp)
target_link_libraries(toml_config PRIVATE fzgmod)
set_target_properties(toml_config PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── cuSZ standalone: LorenzoQuant + Huffman without Pipeline DAG ──────────────
add_executable(cusz_standalone cusz_standalone.cpp)
target_link_libraries(cusz_standalone PRIVATE fzgmod)
set_target_properties(cusz_standalone PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── Standalone Lorenzo code-distribution analyser ─────────────────────────────
add_executable(analyze_lorenzo analyze_lorenzo.cpp)
target_link_libraries(analyze_lorenzo PRIVATE fzgmod)
set_target_properties(analyze_lorenzo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── PREALLOCATE vs MINIMAL memory strategy comparison ─────────────────────────
add_executable(pfpl_memory_strategies pfpl_memory_strategies.cpp)
target_link_libraries(pfpl_memory_strategies PRIVATE fzgmod)
set_target_properties(pfpl_memory_strategies PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── MINIMAL / PREALLOCATE / CUDA Graph capture three-way comparison ───────────
add_executable(pfpl_graph_capture pfpl_graph_capture.cpp)
target_link_libraries(pfpl_graph_capture PRIVATE fzgmod)
set_target_properties(pfpl_graph_capture PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── Getting-started: LorenzoQuant → Bitshuffle → RZE ─────────────────────────
add_executable(lorenzo_intro lorenzo_intro.cpp)
target_link_libraries(lorenzo_intro PRIVATE fzgmod)
set_target_properties(lorenzo_intro PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── Manual stage execution vs Pipeline DAG throughput comparison ──────────────
add_executable(pfpl_manual_vs_dag pfpl_manual_vs_dag.cpp)
target_link_libraries(pfpl_manual_vs_dag PRIVATE fzgmod)
set_target_properties(pfpl_manual_vs_dag PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── File IO: compress to .fzm file and decompress from file ───────────────────
add_executable(file_io_example file_io_example.cpp)
target_link_libraries(file_io_example PRIVATE fzgmod)
set_target_properties(file_io_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── Decode-only slots: in-memory metadata header, no warmup compress() ────────
add_executable(decode_only_slots decode_only_slots.cpp)
target_link_libraries(decode_only_slots PRIVATE fzgmod)
set_target_properties(decode_only_slots PROPERTIES
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

# ── Lorenzo mode comparison: raw-delta vs zigzag code distributions ───────────
add_executable(compare_lorenzo_modes compare_lorenzo_modes.cpp)
target_link_libraries(compare_lorenzo_modes PRIVATE fzgmod)
set_target_properties(compare_lorenzo_modes PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── Huffman vs ANS throughput and compression-ratio comparison ─────────────────
add_executable(cusz_huffman_vs_ans cusz_huffman_vs_ans.cpp)
target_link_libraries(cusz_huffman_vs_ans PRIVATE fzgmod)
set_target_properties(cusz_huffman_vs_ans PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── Predictor comparison: LorenzoQuant vs GInterp (cuSZ-Hi port) ──────────────
add_executable(compare_predictors compare_predictors.cpp)
target_link_libraries(compare_predictors PRIVATE fzgmod)
set_target_properties(compare_predictors PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── Error-bound mode analysis: ABS / NOA / PREL vs exact REL ─────────────────
add_executable(eb_mode_analysis eb_mode_analysis.cpp)
target_link_libraries(eb_mode_analysis PRIVATE fzgmod)
set_target_properties(eb_mode_analysis PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── Lossless back-end comparison: cuSZ Huffman vs FZ-GPU bitplane-RZE ─────────
add_executable(cusz_huffman_vs_bitplane_rze cusz_huffman_vs_bitplane_rze.cpp)
target_link_libraries(cusz_huffman_vs_bitplane_rze PRIVATE fzgmod)
set_target_properties(cusz_huffman_vs_bitplane_rze PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── cuSZp 1/2/3 variants: performance + CUDA graph-mode check ─────────────────
add_executable(cuszp_variants cuszp_variants.cpp)
target_link_libraries(cuszp_variants PRIVATE fzgmod)
set_target_properties(cuszp_variants PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── cuSZ-Hi LC lossless back-end (tp/cr chains + merged-blob via MergeStage) ──
add_executable(lc_lossless_pipeline lc_lossless_pipeline.cpp)
target_link_libraries(lc_lossless_pipeline PRIVATE fzgmod)
set_target_properties(lc_lossless_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── Zigzag vs Negabinary vs TCMS: self-contained integer transform comparison ─
add_executable(integer_transforms_intro integer_transforms_intro.cpp)
set_target_properties(integer_transforms_intro PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── GPU Zstandard: GPULZ split ports -> Huffman/ANS -> Merge ─────────────────
add_executable(gpu_zstd gpu_zstd.cpp)
target_link_libraries(gpu_zstd PRIVATE fzgmod)
set_target_properties(gpu_zstd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── SPERR GPU pipeline + sparse outlier correction: a TRUE pointwise error
#    bound on top of CDF97 -> Quantizer -> SPECK2D. Calls the Stage classes
#    directly (not through Pipeline/DAG) -- see the file's own doc comment
#    and memory/speck_gpu_design.md sec.9 for why (no DAG "Tee" support yet
#    for a stage needing both the raw input and a mid-pipeline buffer).
add_executable(sperr_gpu_bounded sperr_gpu_bounded.cu)
target_link_libraries(sperr_gpu_bounded PRIVATE fzgmod fzgmod_modules fzgmod_mem)
set_target_properties(sperr_gpu_bounded PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")

# ── DAG-integrated version of the above: real Pipeline/DAG, TeeStage +
#    Cdf97OutlierCorrectStage wired as real stages (not manual Stage calls).
add_executable(sperr_gpu_dag_bounded sperr_gpu_dag_bounded.cu)
target_link_libraries(sperr_gpu_dag_bounded PRIVATE fzgmod fzgmod_modules fzgmod_mem)
set_target_properties(sperr_gpu_dag_bounded PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")
