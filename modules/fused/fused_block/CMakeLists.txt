target_sources(fzgmod_modules PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/fused_block.cu
  ${CMAKE_CURRENT_LIST_DIR}/nvrtc_warp_fusion.cu
)
