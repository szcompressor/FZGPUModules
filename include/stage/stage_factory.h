#pragma once

#include "stage/stage.h"
#include "fzm_format.h"
#include "encoders/diff/diff.h"
#include "encoders/RLE/rle.h"
#include "predictors/lorenzo/lorenzo.h"
#include "stage/mock_stages.h"

#include <memory>
#include <stdexcept>
#include <cstring>

namespace fz {

/**
 * Create a stage from its StageType and serialized config
 *
 * Used during decompression to reconstruct the pipeline from the FZM header.
 * The returned stage is heap-allocated; the caller owns the pointer.
 *
 * @param type Stage type from FZMStageInfo
 * @param config Serialized config data (from FZMStageInfo.stage_config or FZMBufferEntry.stage_config)
 * @param config_size Number of valid bytes in config
 * @return Newly created stage (caller owns)
 */
inline Stage* createStage(StageType type, const uint8_t* config, size_t config_size) {
    Stage* stage = nullptr;

    switch (type) {
        case StageType::LORENZO_1D: {
            // Lorenzo needs template params from config
            if (config_size >= sizeof(LorenzoConfig)) {
                LorenzoConfig lc;
                std::memcpy(&lc, config, sizeof(LorenzoConfig));

                // Select template instantiation from stored types
                // Currently only <float, uint16_t> is used in practice
                if (lc.input_type == DataType::FLOAT32 && lc.code_type == DataType::UINT16) {
                    auto* s = new LorenzoStage<float, uint16_t>();
                    s->deserializeHeader(config, config_size);
                    stage = s;
                } else if (lc.input_type == DataType::FLOAT64 && lc.code_type == DataType::UINT16) {
                    auto* s = new LorenzoStage<double, uint16_t>();
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
            // Diff needs element type from config
            if (config_size >= 1) {
                DataType dt;
                std::memcpy(&dt, config, sizeof(DataType));
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
                stage->deserializeHeader(config, config_size);
            } else {
                // No config — default to float
                stage = new DifferenceStage<float>();
            }
            break;
        }

        case StageType::RLE: {
            // RLE needs element type from config
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

        case StageType::PASSTHROUGH: {
            stage = new PassThroughStage();
            break;
        }

        default:
            throw std::runtime_error("Unknown stage type: "
                + std::to_string(static_cast<uint16_t>(type)));
    }

    return stage;
}

} // namespace fz
