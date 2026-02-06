#pragma once

#include "stage/stage.h"
#include <cstdint>

struct OutlierEncodeStage : Stage {
  float* outliers;
  int num_outliers;
  
  uint8_t* encoded;
  size_t encoded_bytes;

  void execute(StageContext& ctx) override;
  void cleanup(StageContext& ctx) override;
};
