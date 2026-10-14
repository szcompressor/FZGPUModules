# Explicit module catalog; source declarations live with each module.

set(FZGMOD_STAGE_MODULES
  predictors/lorenzo
  fused/lorenzo_quant
  fused/adaptive_lorenzo
  quantizers/quantizer
  shufflers/bitshuffle
  coders/rze
  coders/golomb_rice
  coders/rre
  coders/gpulz
  coders/rare
  coders/raze
  coders/clog
  coders/hclog
  structural/tupl
  transforms/log_transform
  structural/merge
  structural/roibin_split
  coders/rle
  predictors/diff
  transforms/zigzag
  transforms/negabinary
  coders/bitpack
  coders/huffman
  coders/ans
  fused/ginterp
  fused/bitplane_rze
  coders/adaptive_bitpack
  predictors/tiled_lorenzo
)

set(FZGMOD_INTERNAL_MODULES
  util/histogram
  fused/fused_block
  fused/chunk_fusion
  fused/common
)

set(FZGMOD_LEGACY_MODULES
)

foreach(module IN LISTS FZGMOD_STAGE_MODULES FZGMOD_INTERNAL_MODULES FZGMOD_LEGACY_MODULES)
  add_subdirectory(${module})
endforeach()
