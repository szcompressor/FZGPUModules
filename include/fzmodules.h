#pragma once

// FZModules - Main API Header

// Format definitions
#include "fzm_format.h"

// Pipeline components
#include "pipeline/compressor.h"
#include "pipeline/dag.h"

// Stage API
#include "stage/stage.h"
#include "stage/mock_stages.h"  // Mock stages for testing

// Real stages
#include "encoders/diff/diff.h"  // Difference coding
#include "predictors/lorenzo/lorenzo.h"  // Lorenzo predictor
