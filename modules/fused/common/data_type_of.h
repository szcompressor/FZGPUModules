#pragma once

/**
 * @file modules/fused/common/data_type_of.h
 * @brief Compile-time C++ type -> DataType enum mapping, shared by the fused
 *        stages that dispatch on multiple template type parameters
 *        (LorenzoQuantStage, GInterpStage, AdaptiveLorenzoStage).
 */

#include "fzm_format.h"

namespace fz {
namespace fused {

/// Primary template intentionally left undefined: instantiating with a type
/// that has no specialization below is a compile error, not a silent
/// fallback to some default DataType.
template <typename T>
constexpr DataType dataTypeOf();

template <> constexpr DataType dataTypeOf<float>()    { return DataType::FLOAT32; }
template <> constexpr DataType dataTypeOf<double>()   { return DataType::FLOAT64; }
template <> constexpr DataType dataTypeOf<int8_t>()   { return DataType::INT8; }
template <> constexpr DataType dataTypeOf<int16_t>()  { return DataType::INT16; }
template <> constexpr DataType dataTypeOf<int32_t>()  { return DataType::INT32; }
template <> constexpr DataType dataTypeOf<int64_t>()  { return DataType::INT64; }
template <> constexpr DataType dataTypeOf<uint8_t>()  { return DataType::UINT8; }
template <> constexpr DataType dataTypeOf<uint16_t>() { return DataType::UINT16; }
template <> constexpr DataType dataTypeOf<uint32_t>() { return DataType::UINT32; }

} // namespace fused
} // namespace fz
