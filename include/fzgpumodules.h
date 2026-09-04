#pragma once

/**
 * @file fzgpumodules.h
 * @brief FZGPUModules main API header — include this to access the full library.
 */

#include <vector>

#include "backend/types.h"
#include "fzm_format.h"

#include "pipeline/compressor.h"
#include "pipeline/stat.h"

// CompressionDAG and the fusion planner/registry are the ADVANCED tier — no
// source-compatibility promise. Pipeline still works without naming them (they are
// pulled transitively by compressor.h), so this umbrella no longer advertises them.
// Reach for them deliberately: #include "advanced/dag.h", "advanced/fusion_*.h".

#include "stage/stage.h"

#include "log.h"

#include "predictors/diff/diff.h"
#include "predictors/lorenzo/lorenzo_stage.h"
#include "predictors/tiled_lorenzo/tiled_lorenzo_stage.h"
#include "coders/rle/rle.h"
#include "coders/bitpack/bitpack_stage.h"
#include "coders/adaptive_bitpack/adaptive_bitpack_stage.h"
#include "coders/rze/rze_stage.h"
#include "coders/rre/rre_stage.h"
#include "coders/rare/rare_stage.h"
#include "coders/raze/raze_stage.h"
#include "coders/clog/clog_stage.h"
#include "coders/hclog/hclog_stage.h"
#include "coders/gpulz/gpulz_stage.h"
#include "coders/speck2d/speck2d_stage.h"
#include "coders/cdf97_outlier_correct/cdf97_outlier_correct_stage.h"
#include "structural/tupl/tupl_stage.h"
#include "structural/merge/merge_stage.h"
#include "fused/adaptive_lorenzo/adaptive_lorenzo_stage.h"
#include "fused/lorenzo_quant/lorenzo_quant.h"
#include "quantizers/quantizer/quantizer.h"
#include "transforms/zigzag/zigzag_stage.h"
#include "transforms/log_transform/log_transform_stage.h"
#include "transforms/cdf97/cdf97_stage.h"
#include "transforms/negabinary/negabinary.h"
#include "transforms/negabinary/negabinary_stage.h"
#include "shufflers/bitshuffle/bitshuffle_stage.h"
#include "coders/huffman/huffman_stage.h"
#include "coders/ans/ans_stage.h"
#include "transforms/adm/adm_stage.h"
#include "fused/ginterp/ginterp_stage.h"
#include "fused/bitplane_rze/bitplane_rze_stage.h"
#include "structural/roibin_split/roibin_split_stage.h"
#include "fused/szx/szx_stage.h"
// SZpStage is quarantined as an experimental/reference compressor and is
// intentionally NOT part of this umbrella header. Existing FZM archives still
// decode (its StageType::SZP factory stays linked); the supported expression of
// the SZp algorithm is examples/presets/szp_composed.toml. To use the reference
// implementation directly, include
// "experimental/reference_compressors/szp/szp_stage.h" explicitly.
