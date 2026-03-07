#pragma once

// FZGPUModules - Main API Header

// Standard library
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>

// CUDA
#include <cuda_runtime.h>

// Format definitions
#include "fzm_format.h"

// Pipeline components
#include "pipeline/compressor.h"
#include "pipeline/dag.h"
#include "pipeline/stat.h"

// Stage API
#include "stage/stage.h"
#include "stage/mock_stages.h"  // Mock stages for testing

// Logging
#include "log.h"

// Real stages
#include "encoders/diff/diff.h"    // Difference coding
#include "encoders/RLE/rle.h"      // Run-length encoding
#include "predictors/lorenzo/lorenzo.h"  // Lorenzo predictor
