/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Adapted for FZGPUModules: all encode/decode logic is defined as kernel
 * templates in the included headers; this translation unit exists to ensure
 * the vendored headers compile cleanly in isolation.
 */
#include "ans/GpuANSEncode.h"
#include "ans/GpuANSDecode.h"
#include "ans/GpuANSCodec.h"

namespace fz { namespace ans {
// All encode/decode kernel templates are defined in the headers above.
}} // namespace fz::ans
