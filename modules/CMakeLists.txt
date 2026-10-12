# Explicit module catalog; source declarations live with each module.

set(FZGMOD_STAGE_MODULES
  predictors/lorenzo
  fused/lorenzo_quant
  fused/adaptive_lorenzo
  fused/quant_adaptive_lorenzo
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
  transforms/cdf97
  coders/speck2d
  coders/cdf97_outlier_correct
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
  fused/szx
)

set(FZGMOD_INTERNAL_MODULES
  util/histogram
  fused/fused_block
  fused/chunk_fusion
  fused/common
)

set(FZGMOD_LEGACY_MODULES
  experimental/reference_compressors/szp
)

foreach(module IN LISTS FZGMOD_STAGE_MODULES FZGMOD_INTERNAL_MODULES FZGMOD_LEGACY_MODULES)
  add_subdirectory(${module})
endforeach()
