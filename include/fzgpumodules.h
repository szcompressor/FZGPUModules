#pragma once

/**
 * @file fzgpumodules.h
 * @brief FZGPUModules main API header — include this to access the full library.
 */

#include <cuda_runtime.h>
#include <vector>

#include "fzm_format.h"

#include "pipeline/compressor.h"
#include "pipeline/dag.h"
#include "pipeline/stat.h"

#include "stage/stage.h"

#include "log.h"

#include "encoders/diff/diff.h"
#include "encoders/RLE/rle.h"
#include "predictors/lorenzo_quant/lorenzo_quant.h"
#include "predictors/quantizer/quantizer.h"
#include "transforms/zigzag/zigzag_stage.h"
#include "transforms/negabinary/negabinary.h"
#include "transforms/negabinary/negabinary_stage.h"
#include "transforms/bitshuffle/bitshuffle_stage.h"
#include "transforms/rze/rze_stage.h"
