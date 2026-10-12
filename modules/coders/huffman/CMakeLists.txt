target_sources(fzgmod_modules PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/phf/hf_buf.cc
  ${CMAKE_CURRENT_LIST_DIR}/phf/hf_hl.cc
  ${CMAKE_CURRENT_LIST_DIR}/phf/hf_bk.cc
  ${CMAKE_CURRENT_LIST_DIR}/phf/hf_bk_impl1.cc
  ${CMAKE_CURRENT_LIST_DIR}/phf/hf_bk_internal.cc
  ${CMAKE_CURRENT_LIST_DIR}/phf/hf_canon.cc
  ${CMAKE_CURRENT_LIST_DIR}/phf/hf_kernels.cu
  ${CMAKE_CURRENT_LIST_DIR}/huffman_stage.cu
)
