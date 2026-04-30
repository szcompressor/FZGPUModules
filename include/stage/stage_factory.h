#pragma once

/**
 * @file stage_factory.h
 * @brief Factory function for reconstructing pipeline stages from serialized FZM headers.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "predictors/diff/diff.h"
#include "coders/rle/rle.h"
#include "fused/lorenzo_quant/lorenzo_quant.h"
#include "predictors/lorenzo/lorenzo_stage.h"
#include "transforms/zigzag/zigzag_stage.h"
#include "transforms/negabinary/negabinary_stage.h"
#include "shufflers/bitshuffle/bitshuffle_stage.h"
#include "coders/rze/rze_stage.h"
#include "coders/bitpack/bitpack_stage.h"

#include <memory>
#include <stdexcept>
#include <cstring>

namespace fz {

/**
 * Reconstruct a Stage from a serialized FZM header. Used by the decompressor
 * to rebuild the inverse pipeline from the file.
 *
 * @param type         Stage type read from `FZMStageInfo`.
 * @param config       Serialized config bytes.
 * @param config_size  Number of valid bytes in `config`.
 * @return Heap-allocated Stage; caller takes ownership.
 */
inline Stage* createStage(StageType type, const uint8_t* config, size_t config_size) {
    Stage* stage = nullptr;

    switch (type) {
        case StageType::LORENZO_QUANT: {
            // Dims are restored by deserializeHeader(); template types come from stored fields.
            if (config_size >= sizeof(LorenzoQuantConfig)) {
                LorenzoQuantConfig lc;
                std::memcpy(&lc, config, sizeof(LorenzoQuantConfig));
                if (lc.input_type == DataType::FLOAT32 && lc.code_type == DataType::UINT16) {
                    auto* s = new LorenzoQuantStage<float, uint16_t>();
                    s->deserializeHeader(config, config_size);
                    stage = s;
                } else if (lc.input_type == DataType::FLOAT64 && lc.code_type == DataType::UINT16) {
                    auto* s = new LorenzoQuantStage<double, uint16_t>();
                    s->deserializeHeader(config, config_size);
                    stage = s;
                } else {
                    throw std::runtime_error(
                        "Unsupported Lorenzo template instantiation: input_type="
                        + std::to_string(static_cast<int>(lc.input_type))
                        + " code_type=" + std::to_string(static_cast<int>(lc.code_type)));
                }
            } else {
                throw std::runtime_error("Lorenzo config too small: " + std::to_string(config_size));
            }
            break;
        }

        case StageType::DIFFERENCE: {
            // Header: [0] TIn DataType, [1] TOut DataType, [2..5] chunk_size.
            // TIn == TOut → same-type (legacy); TIn signed + TOut unsigned → negabinary-fused.
            if (config_size >= 2) {
                DataType tin_dt  = static_cast<DataType>(config[0]);
                DataType tout_dt = static_cast<DataType>(config[1]);
                // Negabinary-fused instantiations
                if      (tin_dt == DataType::INT8  && tout_dt == DataType::UINT8)
                    stage = new DifferenceStage<int8_t,  uint8_t>();
                else if (tin_dt == DataType::INT16 && tout_dt == DataType::UINT16)
                    stage = new DifferenceStage<int16_t, uint16_t>();
                else if (tin_dt == DataType::INT32 && tout_dt == DataType::UINT32)
                    stage = new DifferenceStage<int32_t, uint32_t>();
                else if (tin_dt == DataType::INT64 && tout_dt == DataType::UINT64)
                    stage = new DifferenceStage<int64_t, uint64_t>();
                // Same-type instantiations
                else if (tin_dt == DataType::FLOAT32)  stage = new DifferenceStage<float>();
                else if (tin_dt == DataType::FLOAT64)  stage = new DifferenceStage<double>();
                else if (tin_dt == DataType::UINT8)    stage = new DifferenceStage<uint8_t>();
                else if (tin_dt == DataType::UINT16)   stage = new DifferenceStage<uint16_t>();
                else if (tin_dt == DataType::UINT32)   stage = new DifferenceStage<uint32_t>();
                else if (tin_dt == DataType::INT32)    stage = new DifferenceStage<int32_t>();
                else if (tin_dt == DataType::INT64)    stage = new DifferenceStage<int64_t>();
                else
                    throw std::runtime_error("Unsupported Difference data type: "
                        + std::to_string(static_cast<int>(tin_dt)));
                stage->deserializeHeader(config, config_size);
            } else if (config_size >= 1) {
                // Legacy 1-byte header (same-type only)
                DataType dt = static_cast<DataType>(config[0]);
                switch (dt) {
                    case DataType::FLOAT32:  stage = new DifferenceStage<float>(); break;
                    case DataType::FLOAT64:  stage = new DifferenceStage<double>(); break;
                    case DataType::UINT8:    stage = new DifferenceStage<uint8_t>(); break;
                    case DataType::UINT16:   stage = new DifferenceStage<uint16_t>(); break;
                    case DataType::UINT32:   stage = new DifferenceStage<uint32_t>(); break;
                    case DataType::INT32:    stage = new DifferenceStage<int32_t>(); break;
                    case DataType::INT64:    stage = new DifferenceStage<int64_t>(); break;
                    default:
                        throw std::runtime_error("Unsupported Difference data type: "
                            + std::to_string(static_cast<int>(dt)));
                }
            } else {
                stage = new DifferenceStage<float>();
            }
            break;
        }

        case StageType::RLE: {
            if (config_size >= 1) {
                DataType dt;
                std::memcpy(&dt, config, sizeof(DataType));
                switch (dt) {
                    case DataType::UINT8:    stage = new RLEStage<uint8_t>(); break;
                    case DataType::UINT16:   stage = new RLEStage<uint16_t>(); break;
                    case DataType::UINT32:   stage = new RLEStage<uint32_t>(); break;
                    case DataType::INT32:    stage = new RLEStage<int32_t>(); break;
                    default:
                        throw std::runtime_error("Unsupported RLE data type: "
                            + std::to_string(static_cast<int>(dt)));
                }
                stage->deserializeHeader(config, config_size);
            } else {
                // No config — default to uint16_t
                stage = new RLEStage<uint16_t>();
            }
            break;
        }

        case StageType::ZIGZAG: {
            if (config_size >= 2) {
                DataType tin_dt  = static_cast<DataType>(config[0]);
                DataType tout_dt = static_cast<DataType>(config[1]);
                if      (tin_dt == DataType::INT8  && tout_dt == DataType::UINT8)
                    stage = new ZigzagStage<int8_t,  uint8_t>();
                else if (tin_dt == DataType::INT16 && tout_dt == DataType::UINT16)
                    stage = new ZigzagStage<int16_t, uint16_t>();
                else if (tin_dt == DataType::INT32 && tout_dt == DataType::UINT32)
                    stage = new ZigzagStage<int32_t, uint32_t>();
                else if (tin_dt == DataType::INT64 && tout_dt == DataType::UINT64)
                    stage = new ZigzagStage<int64_t, uint64_t>();
                else
                    throw std::runtime_error(
                        "Unsupported ZigzagStage type pair: TIn="
                        + std::to_string(static_cast<int>(tin_dt))
                        + " TOut=" + std::to_string(static_cast<int>(tout_dt)));
            } else {
                // Default: int32_t → uint32_t
                stage = new ZigzagStage<int32_t, uint32_t>();
            }
            stage->deserializeHeader(config, config_size);
            break;
        }

        case StageType::NEGABINARY: {
            if (config_size >= 2) {
                DataType tin_dt  = static_cast<DataType>(config[0]);
                DataType tout_dt = static_cast<DataType>(config[1]);
                if      (tin_dt == DataType::INT8  && tout_dt == DataType::UINT8)
                    stage = new NegabinaryStage<int8_t,  uint8_t>();
                else if (tin_dt == DataType::INT16 && tout_dt == DataType::UINT16)
                    stage = new NegabinaryStage<int16_t, uint16_t>();
                else if (tin_dt == DataType::INT32 && tout_dt == DataType::UINT32)
                    stage = new NegabinaryStage<int32_t, uint32_t>();
                else if (tin_dt == DataType::INT64 && tout_dt == DataType::UINT64)
                    stage = new NegabinaryStage<int64_t, uint64_t>();
                else
                    throw std::runtime_error(
                        "Unsupported NegabinaryStage type pair: TIn="
                        + std::to_string(static_cast<int>(tin_dt))
                        + " TOut=" + std::to_string(static_cast<int>(tout_dt)));
            } else {
                stage = new NegabinaryStage<int32_t, uint32_t>();
            }
            stage->deserializeHeader(config, config_size);
            break;
        }

        case StageType::BITSHUFFLE: {
            auto* s = new BitshuffleStage();
            s->deserializeHeader(config, config_size);
            stage = s;
            break;
        }

        case StageType::BITPACK: {
            // config[0] holds the DataType of T; use it to pick the instantiation.
            DataType dt = (config_size > 0)
                ? static_cast<DataType>(config[0])
                : DataType::UINT16;
            if      (dt == DataType::UINT8)  stage = new BitpackStage<uint8_t>();
            else if (dt == DataType::UINT16) stage = new BitpackStage<uint16_t>();
            else if (dt == DataType::UINT32) stage = new BitpackStage<uint32_t>();
            else throw std::runtime_error(
                    "Unsupported BitpackStage DataType: "
                    + std::to_string(static_cast<int>(dt)));
            stage->deserializeHeader(config, config_size);
            break;
        }

        case StageType::RZE: {
            auto* s = new RZEStage();
            s->deserializeHeader(config, config_size);
            stage = s;
            break;
        }

        case StageType::LORENZO: {
            DataType dt = (config_size >= sizeof(LorenzoConfig))
                ? static_cast<DataType>(config[0])
                : DataType::INT32;
            if      (dt == DataType::INT8)  stage = new LorenzoStage<int8_t>();
            else if (dt == DataType::INT16) stage = new LorenzoStage<int16_t>();
            else if (dt == DataType::INT32) stage = new LorenzoStage<int32_t>();
            else if (dt == DataType::INT64) stage = new LorenzoStage<int64_t>();
            else throw std::runtime_error(
                    "Unsupported LorenzoStage DataType: "
                    + std::to_string(static_cast<int>(dt)));
            stage->deserializeHeader(config, config_size);
            break;
        }

        default:
            throw std::runtime_error("Unknown stage type: "
                + std::to_string(static_cast<uint16_t>(type)));
    }

    return stage;
}

} // namespace fz
