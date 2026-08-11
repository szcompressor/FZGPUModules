#pragma once

/**
 * @file stage_factory.h
 * @brief Factory function for reconstructing pipeline stages from serialized FZM headers.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "predictors/diff/diff.h"
#include "coders/rle/rle.h"
#include "fused/adaptive_lorenzo/adaptive_lorenzo_stage.h"
#include "fused/lorenzo_quant/lorenzo_quant.h"
#include "predictors/lorenzo/lorenzo_stage.h"
#include "transforms/zigzag/zigzag_stage.h"
#include "transforms/negabinary/negabinary_stage.h"
#include "shufflers/bitshuffle/bitshuffle_stage.h"
#include "coders/rze/rze_stage.h"
#include "coders/rre/rre_stage.h"
#include "coders/gpulz/gpulz_stage.h"
#include "coders/rare/rare_stage.h"
#include "coders/raze/raze_stage.h"
#include "coders/clog/clog_stage.h"
#include "coders/hclog/hclog_stage.h"
#include "shufflers/tupl/tupl_stage.h"
#include "structural/merge/merge_stage.h"
#include "structural/roibin_split/roibin_split_stage.h"
#include "coders/bitpack/bitpack_stage.h"
#include "coders/adaptive_bitpack/adaptive_bitpack_stage.h"
#include "predictors/tiled_lorenzo/tiled_lorenzo_stage.h"
#include "coders/huffman/huffman_stage.h"
#include "coders/ans/ans_stage.h"
#include "transforms/adm/adm_stage.h"
#include "fused/ginterp/ginterp_stage.h"
#include "fused/bitplane_rze/bitplane_rze_stage.h"
#include "quantizers/quantizer/quantizer.h"
#include "transforms/log_transform/log_transform_stage.h"

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
            // Header: [0] TIn DataType, [1] TOut DataType, [2..5] chunk_size, [6] FusionMode (fused only).
            // TIn == TOut → same-type (legacy); TIn signed + TOut unsigned → fused
            // (byte 6 selects NEGABINARY (default, for 6-byte legacy headers) vs. ZIGZAG).
            if (config_size >= 2) {
                DataType tin_dt  = static_cast<DataType>(config[0]);
                DataType tout_dt = static_cast<DataType>(config[1]);
                FusionMode mode = FusionMode::NEGABINARY;
                if (config_size >= 7) mode = static_cast<FusionMode>(config[6]);
                // Negabinary/zigzag-fused instantiations
                if (tin_dt == DataType::INT8 && tout_dt == DataType::UINT8) {
                    stage = (mode == FusionMode::ZIGZAG)
                        ? static_cast<Stage*>(new DifferenceStage<int8_t, uint8_t, FusionMode::ZIGZAG>())
                        : static_cast<Stage*>(new DifferenceStage<int8_t, uint8_t, FusionMode::NEGABINARY>());
                } else if (tin_dt == DataType::INT16 && tout_dt == DataType::UINT16) {
                    stage = (mode == FusionMode::ZIGZAG)
                        ? static_cast<Stage*>(new DifferenceStage<int16_t, uint16_t, FusionMode::ZIGZAG>())
                        : static_cast<Stage*>(new DifferenceStage<int16_t, uint16_t, FusionMode::NEGABINARY>());
                } else if (tin_dt == DataType::INT32 && tout_dt == DataType::UINT32) {
                    stage = (mode == FusionMode::ZIGZAG)
                        ? static_cast<Stage*>(new DifferenceStage<int32_t, uint32_t, FusionMode::ZIGZAG>())
                        : static_cast<Stage*>(new DifferenceStage<int32_t, uint32_t, FusionMode::NEGABINARY>());
                } else if (tin_dt == DataType::INT64 && tout_dt == DataType::UINT64) {
                    stage = (mode == FusionMode::ZIGZAG)
                        ? static_cast<Stage*>(new DifferenceStage<int64_t, uint64_t, FusionMode::ZIGZAG>())
                        : static_cast<Stage*>(new DifferenceStage<int64_t, uint64_t, FusionMode::NEGABINARY>());
                }
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

        case StageType::QUANTIZER: {
            if (config_size < sizeof(QuantizerConfig)) {
                throw std::runtime_error(
                    "QuantizerConfig too small: " + std::to_string(config_size));
            }
            QuantizerConfig qc;
            std::memcpy(&qc, config, sizeof(QuantizerConfig));
            if (qc.input_type == DataType::FLOAT32 && qc.code_type == DataType::UINT16) {
                auto* s = new QuantizerStage<float, uint16_t>();
                s->deserializeHeader(config, config_size);
                stage = s;
            } else if (qc.input_type == DataType::FLOAT32 && qc.code_type == DataType::UINT32) {
                auto* s = new QuantizerStage<float, uint32_t>();
                s->deserializeHeader(config, config_size);
                stage = s;
            } else if (qc.input_type == DataType::FLOAT64 && qc.code_type == DataType::UINT16) {
                auto* s = new QuantizerStage<double, uint16_t>();
                s->deserializeHeader(config, config_size);
                stage = s;
            } else if (qc.input_type == DataType::FLOAT64 && qc.code_type == DataType::UINT32) {
                auto* s = new QuantizerStage<double, uint32_t>();
                s->deserializeHeader(config, config_size);
                stage = s;
            } else {
                throw std::runtime_error(
                    "Unsupported QuantizerStage types: input_type="
                    + std::to_string(static_cast<int>(qc.input_type))
                    + " code_type=" + std::to_string(static_cast<int>(qc.code_type)));
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
                    case DataType::UINT64:   stage = new RLEStage<uint64_t>(); break;
                    case DataType::INT8:     stage = new RLEStage<int8_t>(); break;
                    case DataType::INT16:    stage = new RLEStage<int16_t>(); break;
                    case DataType::INT32:    stage = new RLEStage<int32_t>(); break;
                    case DataType::INT64:    stage = new RLEStage<int64_t>(); break;
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

        case StageType::RRE: {
            auto* s = new RREStage();
            s->deserializeHeader(config, config_size);
            stage = s;
            break;
        }

        case StageType::GPULZ: {
            auto* s = new GPULZStage();
            s->deserializeHeader(config, config_size);
            stage = s;
            break;
        }

        case StageType::RARE: {
            auto* s = new RAREStage();
            s->deserializeHeader(config, config_size);
            stage = s;
            break;
        }

        case StageType::RAZE: {
            auto* s = new RAZEStage();
            s->deserializeHeader(config, config_size);
            stage = s;
            break;
        }

        case StageType::CLOG: {
            auto* s = new CLOGStage();
            s->deserializeHeader(config, config_size);
            stage = s;
            break;
        }

        case StageType::HCLOG: {
            auto* s = new HCLOGStage();
            s->deserializeHeader(config, config_size);
            stage = s;
            break;
        }

        case StageType::TUPL: {
            auto* s = new TUPLStage();
            s->deserializeHeader(config, config_size);
            stage = s;
            break;
        }

        case StageType::LOG_TRANSFORM: {
            auto* s = new LogTransformStage<float>();
            s->deserializeHeader(config, config_size);
            stage = s;
            break;
        }

        case StageType::ADAPTIVE_LORENZO: {
            // Element type is stored in the config's first byte.
            DataType dt = (config_size > 0) ? static_cast<DataType>(config[0])
                                            : DataType::INT32;
            if (dt == DataType::INT16) {
                auto* s = new AdaptiveLorenzoStage<int16_t>();
                s->deserializeHeader(config, config_size);
                stage = s;
            } else {
                auto* s = new AdaptiveLorenzoStage<int32_t>();
                s->deserializeHeader(config, config_size);
                stage = s;
            }
            break;
        }

        case StageType::MERGE: {
            auto* s = new MergeStage();
            s->deserializeHeader(config, config_size);
            stage = s;
            break;
        }

        case StageType::ROIBIN_SPLIT: {
            // Byte 20 of the header holds the field DataType
            // (16 B of dims+npeaks, then uint16 hw, uint16 bin).
            DataType dt = (config_size > 20)
                ? static_cast<DataType>(config[20])
                : DataType::FLOAT32;
            if      (dt == DataType::FLOAT32) stage = new ROIBinSplitStage<float>();
            else if (dt == DataType::FLOAT64) stage = new ROIBinSplitStage<double>();
            else throw std::runtime_error(
                    "Unsupported ROIBinSplitStage DataType: "
                    + std::to_string(static_cast<int>(dt)));
            stage->deserializeHeader(config, config_size);
            break;
        }

        case StageType::HUFFMAN: {
            // config[0] holds the DataType of T; use it to pick the instantiation.
            DataType dt = (config_size > 0)
                ? static_cast<DataType>(config[0])
                : DataType::UINT16;
            if      (dt == DataType::UINT8)  stage = new HuffmanStage<uint8_t>();
            else if (dt == DataType::UINT16) stage = new HuffmanStage<uint16_t>();
            else if (dt == DataType::UINT32) stage = new HuffmanStage<uint32_t>();
            else throw std::runtime_error(
                    "Unsupported HuffmanStage DataType: "
                    + std::to_string(static_cast<int>(dt)));
            stage->deserializeHeader(config, config_size);
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

        case StageType::ANS: {
#if !defined(FZGMOD_BACKEND_HIP) && !defined(FZGMOD_BACKEND_SYCL)
            auto* s = new ANSStage();
            s->deserializeHeader(config, config_size);
            stage = s;
#else
            // Unlike Pipeline::addStage<T>() (see Stage::isSupportedOnBackend()'s
            // doc comment), this switch is not a template, so `if constexpr`
            // can't gate the `new ANSStage()` above -- needs its own #if to
            // avoid referencing ANSStage's constructor, whose .cu translation
            // unit isn't compiled on this backend.
            throw std::runtime_error(
                "deserializeStage(): 'ANS' stage is not supported on the "
                "current GPU backend");
#endif
            break;
        }

        case StageType::ADM: {
            auto* s = new ADMStage();
            s->deserializeHeader(config, config_size);
            stage = s;
            break;
        }

        case StageType::G_INTERP: {
            // Code type stored in config[5] (DataType code_type in GInterpConfig).
            if (config_size < sizeof(GInterpConfig)) {
                throw std::runtime_error(
                    "GInterp config too small: " + std::to_string(config_size));
            }
            GInterpConfig gc;
            std::memcpy(&gc, config, sizeof(GInterpConfig));
            auto make_ginterp = [&](auto input_tag) {
                using TInput = decltype(input_tag);
                if (gc.code_type == DataType::UINT8) {
                    auto* s = new GInterpStage<TInput, uint8_t>();
                    s->deserializeHeader(config, config_size);
                    stage = s;
                } else if (gc.code_type == DataType::UINT16) {
                    auto* s = new GInterpStage<TInput, uint16_t>();
                    s->deserializeHeader(config, config_size);
                    stage = s;
                } else if (gc.code_type == DataType::UINT32) {
                    auto* s = new GInterpStage<TInput, uint32_t>();
                    s->deserializeHeader(config, config_size);
                    stage = s;
                } else {
                    throw std::runtime_error(
                        "Unsupported GInterp code_type: "
                        + std::to_string(static_cast<int>(gc.code_type)));
                }
            };
            if (gc.input_type == DataType::FLOAT32) {
                make_ginterp(float{});
            } else if (gc.input_type == DataType::FLOAT64) {
                make_ginterp(double{});
            } else {
                throw std::runtime_error(
                    "Unsupported GInterp input_type: "
                    + std::to_string(static_cast<int>(gc.input_type)));
            }
            break;
        }

        case StageType::BITPLANE_RZE: {
            auto* s = new BitplaneRZEStage();
            s->deserializeHeader(config, config_size);
            stage = s;
            break;
        }

        case StageType::ADAPTIVE_BITPACK: {
            // config[0] holds the DataType of T (INT16 / INT32).
            DataType dt = (config_size > 0)
                ? static_cast<DataType>(config[0])
                : DataType::INT32;
            if      (dt == DataType::INT16) stage = new AdaptiveBitpackStage<int16_t>();
            else if (dt == DataType::INT32) stage = new AdaptiveBitpackStage<int32_t>();
            else throw std::runtime_error(
                    "Unsupported AdaptiveBitpackStage DataType: "
                    + std::to_string(static_cast<int>(dt)));
            stage->deserializeHeader(config, config_size);
            break;
        }

        case StageType::TILED_LORENZO: {
            // config[0] holds the DataType of T (INT16 / INT32).
            DataType dt = (config_size > 0)
                ? static_cast<DataType>(config[0])
                : DataType::INT32;
            if      (dt == DataType::INT16) stage = new TiledLorenzoStage<int16_t>();
            else if (dt == DataType::INT32) stage = new TiledLorenzoStage<int32_t>();
            else throw std::runtime_error(
                    "Unsupported TiledLorenzoStage DataType: "
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
