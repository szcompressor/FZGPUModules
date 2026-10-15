target_sources(fzgmod_modules PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/nvrtc_jit.cpp
)

# Embed JIT headers and propagate driver linkage; see docs/codebase_notes.md CN-BUILD-2.
if(NOT FZGMOD_BACKEND STREQUAL "HIP")
  target_link_libraries(fzgmod_modules PRIVATE CUDA::nvrtc PUBLIC CUDA::cuda_driver)
  set(FZGMOD_NVRTC_HEADERS "${PROJECT_BINARY_DIR}/generated/fz_nvrtc_headers.h")
  # Unchanged output preserves its timestamp.
  add_custom_target(fzgmod_nvrtc_headers ALL
    COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/scripts/gen_nvrtc_headers.py"
            --out "${FZGMOD_NVRTC_HEADERS}"
    BYPRODUCTS "${FZGMOD_NVRTC_HEADERS}"
    COMMENT "Embedding NVRTC device headers"
    VERBATIM
  )
  add_dependencies(fzgmod_modules fzgmod_nvrtc_headers)
  target_include_directories(fzgmod_modules PRIVATE "${PROJECT_BINARY_DIR}/generated")
endif()
