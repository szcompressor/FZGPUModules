#pragma once

// Pipeline infrastructure
#include "pipeline/compressor.h"
#include "pipeline/dag.h"
#include "stage/stage.h"

// Predictors
#include "predictors/lorenzo/lorenzo.h"

// Encoding
#include "encoding/bitpacking/bitpacking.h"
#include "encoding/run_length_encoding/RLE.h"
#include "encoding/concatenation/concat.h"
#include "encoding/difference/diff_encode.h"
