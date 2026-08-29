/**
 * @file config.cpp
 * @brief TOML config file load/save for Pipeline.
 *
 * toml++ is only included here — it never leaks into public headers.
 */

// nvc++ defines __clang__ (it's LLVM-based), so toml++ enables Clang-specific
// attributes in Release/NDEBUG builds that nvc++'s optimizer mishandles:
//
//   TOML_PURE  = __attribute__((pure))   — applied to key::operator< and all
//                key comparison operators; nvc++ with -fast may cache/elide the
//                string reads, leaving one operand uninitialized → segfault in
//                std::string::compare during std::map BST traversal.
//   TOML_CONST = __attribute__((const))  — even stronger; same risk.
//   TOML_ALWAYS_INLINE = __attribute__((__always_inline__)) — amplifies the above
//                by forcing inlining of the broken comparison path.
//   TOML_ASSUME(expr) = __builtin_assume(expr) — nvc++ optimizer uses the hint
//                to eliminate null/bounds guards → UB → segfault in parser.
//
// Override all four to safe no-ops before the toml++ include.
#if defined(__NVCOMPILER)
#  define TOML_ASSUME(expr)   static_cast<void>(0)
#  define TOML_PURE
#  define TOML_CONST
#  define TOML_ALWAYS_INLINE  inline
#endif
#define TOML_HEADER_ONLY 1
#include <toml++/toml.hpp>

#include "pipeline/compressor.h"
#include "pipeline/config.h"
// Generated at build time by scripts/gen_stage_fingerprints.py (see CMakeLists.txt).
#include "fz_stage_fingerprints.h"

// All stage types supported by loadConfig / saveConfig
#include "fused/lorenzo_quant/lorenzo_quant.h"
#include "predictors/lorenzo/lorenzo_stage.h"
#include "predictors/tiled_lorenzo/tiled_lorenzo_stage.h"
#include "quantizers/quantizer/quantizer.h"
#include "shufflers/bitshuffle/bitshuffle_stage.h"
#include "coders/rze/rze_stage.h"
#include "coders/rre/rre_stage.h"
#include "coders/gpulz/gpulz_stage.h"
#include "coders/rare/rare_stage.h"
#include "coders/raze/raze_stage.h"
#include "coders/clog/clog_stage.h"
#include "coders/hclog/hclog_stage.h"
#include "shufflers/tupl/tupl_stage.h"
#include "transforms/log_transform/log_transform_stage.h"
#include "transforms/cdf97/cdf97_stage.h"
#include "coders/speck2d/speck2d_stage.h"
#include "coders/cdf97_outlier_correct/cdf97_outlier_correct_stage.h"
#include "structural/merge/merge_stage.h"
#include "structural/roibin_split/roibin_split_stage.h"
#include "fused/szx/szx_stage.h"
#include "fused/szp/szp_stage.h"
#include "transforms/zigzag/zigzag_stage.h"
#include "transforms/negabinary/negabinary_stage.h"
#include "coders/bitpack/bitpack_stage.h"
#include "coders/adaptive_bitpack/adaptive_bitpack_stage.h"
#include "coders/huffman/huffman_stage.h"
#include "coders/ans/ans_stage.h"
#include "transforms/adm/adm_stage.h"
#include "coders/rle/rle.h"
#include "predictors/diff/diff.h"
#include "fused/ginterp/ginterp_stage.h"
#include "fused/bitplane_rze/bitplane_rze_stage.h"
#include "fused/adaptive_lorenzo/adaptive_lorenzo_stage.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

// ─────────────────────────────────────────────────────────────────────────────
// Small string-conversion helpers (local to this TU)
// ─────────────────────────────────────────────────────────────────────────────

static MemoryStrategy strategyFromString(const std::string& s) {
    if (s == "PREALLOCATE") return MemoryStrategy::PREALLOCATE;
    if (s == "MINIMAL")     return MemoryStrategy::MINIMAL;
    throw std::runtime_error("loadConfig: unknown memory_strategy \"" + s + "\"");
}

static std::string strategyToString(MemoryStrategy s) {
    return s == MemoryStrategy::PREALLOCATE ? "PREALLOCATE" : "MINIMAL";
}

static ErrorBoundMode ebModeFromString(const std::string& s) {
    if (s == "ABS")  return ErrorBoundMode::ABS;
    if (s == "REL")  return ErrorBoundMode::REL;
    if (s == "NOA")  return ErrorBoundMode::NOA;
    if (s == "PREL") return ErrorBoundMode::PREL;
    throw std::runtime_error("loadConfig: unknown error_bound_mode \"" + s +
                             "\" (expected ABS|REL|NOA|PREL)");
}

static std::string ebModeToString(ErrorBoundMode m) {
    switch (m) {
        case ErrorBoundMode::REL:  return "REL";
        case ErrorBoundMode::NOA:  return "NOA";
        case ErrorBoundMode::PREL: return "PREL";
        default:                   return "ABS";
    }
}

static std::string tomlEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\\' || c == '"') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

static HuffmanBookModel huffmanBookModelFromString(const std::string& s) {
    if (s == "Gaussian")          return HuffmanBookModel::Gaussian;
    if (s == "Laplace")           return HuffmanBookModel::Laplace;
    if (s == "GeneralizedNormal") return HuffmanBookModel::GeneralizedNormal;
    if (s == "Uniform")           return HuffmanBookModel::Uniform;
    throw std::runtime_error("loadConfig: unknown Huffman book_model \"" + s + "\"");
}

static const char* huffmanBookModelToString(HuffmanBookModel m) {
    switch (m) {
        case HuffmanBookModel::Laplace:           return "Laplace";
        case HuffmanBookModel::GeneralizedNormal: return "GeneralizedNormal";
        case HuffmanBookModel::Uniform:           return "Uniform";
        case HuffmanBookModel::Gaussian:
        default:                                  return "Gaussian";
    }
}

static HuffmanExecutionMode huffmanExecutionModeFromString(const std::string& s) {
    if (s == "HostCoordinated") return HuffmanExecutionMode::HostCoordinated;
    if (s == "DeviceResident")  return HuffmanExecutionMode::DeviceResident;
    throw std::runtime_error(
        "loadConfig: unknown Huffman execution_mode \"" + s + "\"");
}

static const char* huffmanExecutionModeToString(HuffmanExecutionMode mode) {
    return mode == HuffmanExecutionMode::DeviceResident
        ? "DeviceResident" : "HostCoordinated";
}

static DataType dataTypeFromString(const std::string& s) {
    if (s == "float32") return DataType::FLOAT32;
    if (s == "float64") return DataType::FLOAT64;
    if (s == "uint8")   return DataType::UINT8;
    if (s == "uint16")  return DataType::UINT16;
    if (s == "uint32")  return DataType::UINT32;
    if (s == "uint64")  return DataType::UINT64;
    if (s == "int8")    return DataType::INT8;
    if (s == "int16")   return DataType::INT16;
    if (s == "int32")   return DataType::INT32;
    if (s == "int64")   return DataType::INT64;
    throw std::runtime_error("loadConfig: unknown data type \"" + s + "\"");
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage parsing helpers (load direction)
// ─────────────────────────────────────────────────────────────────────────────

// Reads an optional string key from a TOML table, returns default if absent.
static std::string optStr(const toml::table& t, std::string_view key, std::string def = "") {
    if (auto v = t[key].as_string()) return v->get();
    return def;
}
static int64_t optInt(const toml::table& t, std::string_view key, int64_t def = 0) {
    if (auto v = t[key].as_integer()) return v->get();
    return def;
}
static double optDbl(const toml::table& t, std::string_view key, double def = 0.0) {
    if (auto v = t[key].as_floating_point()) return v->get();
    if (auto v = t[key].as_integer())        return static_cast<double>(v->get());
    return def;
}
static bool optBool(const toml::table& t, std::string_view key, bool def = false) {
    if (auto v = t[key].as_boolean()) return v->get();
    return def;
}

// Add a plain integer Lorenzo predictor stage (dispatches on data_type string).
static Stage* addLorenzoStage(Pipeline& p, const toml::table& t) {
    std::string dt_str = optStr(t, "data_type", "int32");
    DataType dt = dataTypeFromString(dt_str);

    // Both are constructor arguments, not post-add setters: centering adds a
    // "means" port and addStage() captures the port count at add-time.
    const auto bs   = static_cast<uint32_t>(optInt(t, "block_size", 0));
    const bool cent = optBool(t, "centering", false);
    const auto ord  = static_cast<uint8_t>(optInt(t, "order", 1));
    if (cent && bs == 0)
        throw std::runtime_error(
            "Lorenzo stage: centering = true requires block_size > 0");
    if (ord == 2 && bs == 0)
        throw std::runtime_error(
            "Lorenzo stage: order = 2 requires block_size > 0");

    Stage* s = nullptr;
    auto configure = [&](auto* lrz) { s = lrz; };

    if      (dt == DataType::INT8)  configure(p.addStage<LorenzoStage<int8_t>>(bs, cent, ord));
    else if (dt == DataType::INT16) configure(p.addStage<LorenzoStage<int16_t>>(bs, cent, ord));
    else if (dt == DataType::INT32) configure(p.addStage<LorenzoStage<int32_t>>(bs, cent, ord));
    else if (dt == DataType::INT64) configure(p.addStage<LorenzoStage<int64_t>>(bs, cent, ord));
    else
        throw std::runtime_error(
            "loadConfig: unsupported Lorenzo data_type \"" + dt_str + "\"");

    return s;
}

// Add an AdaptiveLorenzo stage (per-tile adaptive multi-order + centering).
static Stage* addAdaptiveLorenzoStage(Pipeline& p, const toml::table& t) {
    DataType dt = dataTypeFromString(optStr(t, "data_type", "int32"));

    // All four knobs are constructor arguments: the stage's port count and tile
    // geometry are both fixed at add-time.
    auto make = [&](auto* tag) {
        using StageT = std::remove_pointer_t<decltype(tag)>;
        typename StageT::Config c;
        c.coder_block_size = static_cast<uint32_t>(optInt(t, "coder_block_size", 32));
        c.blocks_per_tile  = static_cast<uint32_t>(optInt(t, "blocks_per_tile", 8));
        c.enable_order2    = optBool(t, "enable_order2", true);
        c.enable_centering = optBool(t, "enable_centering", true);
        return p.addStage<StageT>(c);
    };

    if (dt == DataType::INT16)
        return make(static_cast<AdaptiveLorenzoStage<int16_t>*>(nullptr));
    if (dt == DataType::INT32)
        return make(static_cast<AdaptiveLorenzoStage<int32_t>*>(nullptr));
    throw std::runtime_error(
        "loadConfig: AdaptiveLorenzo supports data_type \"int16\" or \"int32\", got \""
        + optStr(t, "data_type", "int32") + "\"");
}

static void saveAdaptiveLorenzoStage(Stage* s, std::ostringstream& out) {
    uint8_t buf[sizeof(AdaptiveLorenzoConfig)] = {};
    if (s->serializeHeader(0, buf, sizeof(buf)) >= sizeof(AdaptiveLorenzoConfig)) {
        AdaptiveLorenzoConfig c;
        std::memcpy(&c, buf, sizeof(c));
        out << "data_type = \"" << dataTypeToString(c.data_type) << "\"\n";
        out << "blocks_per_tile = " << static_cast<int>(c.blocks_per_tile) << "\n";
        out << "enable_order2 = " << (c.enable_order2 ? "true" : "false") << "\n";
        out << "enable_centering = " << (c.enable_centering ? "true" : "false") << "\n";
    }
}

// Add a LorenzoQuant stage (dispatches on input_type / code_type strings).
static Stage* addLorenzoQuantStage(Pipeline& p, const toml::table& t) {
    std::string in_type   = optStr(t, "input_type", "float32");
    std::string code_type = optStr(t, "code_type",  "uint16");

    DataType in_dt   = dataTypeFromString(in_type);
    DataType code_dt = dataTypeFromString(code_type);

    Stage* s = nullptr;

    // Centering must be a *constructor* argument: it adds a "means" port and
    // addStage() captures the port count at add-time. Every other key below is
    // a plain setter because none of them change the port layout.
    const bool cent = optBool(t, "centering", false);
    auto add = [&](auto* tag) {
        using StageT = std::remove_pointer_t<decltype(tag)>;
        typename StageT::Config c;
        c.centering = cent;
        return p.addStage<StageT>(c);
    };

    auto configure = [&](auto* lrz) {
        lrz->setErrorBound(static_cast<float>(optDbl(t, "error_bound", 1e-3)));
        lrz->setErrorBoundMode(ebModeFromString(optStr(t, "error_bound_mode", "ABS")));
        lrz->setQuantRadius(static_cast<int>(optInt(t, "quant_radius", 32768)));
        lrz->setOutlierCapacity(static_cast<float>(optDbl(t, "outlier_capacity", 0.2)));
        lrz->setZigzagCodes(optBool(t, "zigzag_codes", false));
        s = lrz;
    };

    if (in_dt == DataType::FLOAT32 && code_dt == DataType::UINT16)
        configure(add(static_cast<LorenzoQuantStage<float, uint16_t>*>(nullptr)));
    else if (in_dt == DataType::FLOAT64 && code_dt == DataType::UINT16)
        configure(add(static_cast<LorenzoQuantStage<double, uint16_t>*>(nullptr)));
    else if (in_dt == DataType::FLOAT32 && code_dt == DataType::UINT8)
        configure(add(static_cast<LorenzoQuantStage<float, uint8_t>*>(nullptr)));
    else if (in_dt == DataType::FLOAT64 && code_dt == DataType::UINT32)
        configure(add(static_cast<LorenzoQuantStage<double, uint32_t>*>(nullptr)));
    else
        throw std::runtime_error(
            "loadConfig: unsupported Lorenzo type combination input_type=\""
            + in_type + "\" code_type=\"" + code_type + "\"");

    return s;
}

static Stage* addQuantizerStage(Pipeline& p, const toml::table& t) {
    std::string in_type   = optStr(t, "input_type", "float32");
    std::string code_type = optStr(t, "code_type",  "uint32");

    DataType in_dt   = dataTypeFromString(in_type);
    DataType code_dt = dataTypeFromString(code_type);

    Stage* s = nullptr;

    bool linear = optBool(t, "linear_mode", false);
    auto configure = [&](auto* quant) {
        quant->setErrorBound(optDbl(t, "error_bound", 1e-3));
        quant->setErrorBoundMode(ebModeFromString(optStr(t, "error_bound_mode", "REL")));
        quant->setQuantRadius(static_cast<int>(optInt(t, "quant_radius", 32768)));
        quant->setOutlierCapacity(static_cast<float>(optDbl(t, "outlier_capacity", 0.05)));
        // Linear mode produces raw signed codes — zigzag is incompatible, so it
        // defaults off when linear mode is on.
        quant->setZigzagCodes(optBool(t, "zigzag_codes", !linear));

        float threshold = static_cast<float>(optDbl(t, "outlier_threshold", std::numeric_limits<float>::infinity()));
        if (std::isfinite(threshold)) {
            quant->setOutlierThreshold(threshold);
        }

        quant->setInplaceOutliers(optBool(t, "inplace_outliers", false));
        quant->setLinearMode(linear);
        quant->setLinearHighPrecision(optBool(t, "linear_high_precision", false));
        quant->setPowerOfTwoBound(optBool(t, "power_of_two_bound", false));
        quant->setDither(optBool(t, "dither", false));
        quant->setDitherSeed(static_cast<uint64_t>(optInt(t, "dither_seed", 0)));
        quant->setDitherStrength(static_cast<float>(optDbl(t, "dither_strength", 1.0)));
        s = quant;
    };

    if (in_dt == DataType::FLOAT32 && code_dt == DataType::UINT16)
        configure(p.addStage<QuantizerStage<float, uint16_t>>());
    else if (in_dt == DataType::FLOAT32 && code_dt == DataType::UINT32)
        configure(p.addStage<QuantizerStage<float, uint32_t>>());
    else if (in_dt == DataType::FLOAT64 && code_dt == DataType::UINT16)
        configure(p.addStage<QuantizerStage<double, uint16_t>>());
    else if (in_dt == DataType::FLOAT64 && code_dt == DataType::UINT32)
        configure(p.addStage<QuantizerStage<double, uint32_t>>());
    else
        throw std::runtime_error(
            "loadConfig: unsupported Quantizer type combination input_type=\""
            + in_type + "\" code_type=\"" + code_type + "\"");

    return s;
}

static Stage* addRLEStage(Pipeline& p, const toml::table& t) {
    DataType dt = dataTypeFromString(optStr(t, "data_type", "uint16"));
    // 0 (the default) keeps the whole-array path.
    const size_t cs = static_cast<size_t>(optInt(t, "chunk_size", 0));
    auto add = [&](auto* s) { s->setChunkSize(cs); return s; };
    switch (dt) {
        case DataType::UINT8:  return add(p.addStage<RLEStage<uint8_t>>());
        case DataType::UINT16: return add(p.addStage<RLEStage<uint16_t>>());
        case DataType::UINT32: return add(p.addStage<RLEStage<uint32_t>>());
        case DataType::UINT64: return add(p.addStage<RLEStage<uint64_t>>());
        case DataType::INT8:   return add(p.addStage<RLEStage<int8_t>>());
        case DataType::INT16:  return add(p.addStage<RLEStage<int16_t>>());
        case DataType::INT32:  return add(p.addStage<RLEStage<int32_t>>());
        case DataType::INT64:  return add(p.addStage<RLEStage<int64_t>>());
        default:
            throw std::runtime_error("loadConfig: unsupported RLE data_type \""
                + optStr(t, "data_type", "uint16") + "\"");
    }
}

static Stage* addDifferenceStage(Pipeline& p, const toml::table& t) {
    DataType in_dt  = dataTypeFromString(optStr(t, "input_type",  "float32"));
    std::string out_str = optStr(t, "output_type", "");
    DataType out_dt = out_str.empty() ? in_dt : dataTypeFromString(out_str);

    // Same-type instantiations
    if (in_dt == out_dt) {
        switch (in_dt) {
            case DataType::FLOAT32: { auto* s = p.addStage<DifferenceStage<float>>();    s->setChunkSize(optInt(t,"chunk_size",0)); return s; }
            case DataType::FLOAT64: { auto* s = p.addStage<DifferenceStage<double>>();   s->setChunkSize(optInt(t,"chunk_size",0)); return s; }
            case DataType::UINT8:   { auto* s = p.addStage<DifferenceStage<uint8_t>>();  s->setChunkSize(optInt(t,"chunk_size",0)); return s; }
            case DataType::UINT16:  { auto* s = p.addStage<DifferenceStage<uint16_t>>(); s->setChunkSize(optInt(t,"chunk_size",0)); return s; }
            case DataType::UINT32:  { auto* s = p.addStage<DifferenceStage<uint32_t>>(); s->setChunkSize(optInt(t,"chunk_size",0)); return s; }
            case DataType::INT32:   { auto* s = p.addStage<DifferenceStage<int32_t>>();  s->setChunkSize(optInt(t,"chunk_size",0)); return s; }
            case DataType::INT64:   { auto* s = p.addStage<DifferenceStage<int64_t>>();  s->setChunkSize(optInt(t,"chunk_size",0)); return s; }
            default:
                throw std::runtime_error("loadConfig: unsupported Difference input_type");
        }
    }
    // Negabinary/zigzag-fused instantiations (signed → unsigned of same width).
    // fusion_mode = "negabinary" (default, LC's DIFFNB) or "zigzag" (LC's DIFFMS).
    size_t cs = static_cast<size_t>(optInt(t, "chunk_size", 0));
    std::string fusion_str = optStr(t, "fusion_mode", "negabinary");
    FusionMode mode;
    if      (fusion_str == "negabinary") mode = FusionMode::NEGABINARY;
    else if (fusion_str == "zigzag")     mode = FusionMode::ZIGZAG;
    else throw std::runtime_error("loadConfig: unsupported Difference fusion_mode \"" + fusion_str + "\" (expected \"negabinary\" or \"zigzag\")");

    if (in_dt == DataType::INT8 && out_dt == DataType::UINT8) {
        if (mode == FusionMode::ZIGZAG) { auto* s = p.addStage<DifferenceStage<int8_t, uint8_t, FusionMode::ZIGZAG>>();     s->setChunkSize(cs); return s; }
        else                            { auto* s = p.addStage<DifferenceStage<int8_t, uint8_t, FusionMode::NEGABINARY>>(); s->setChunkSize(cs); return s; }
    }
    if (in_dt == DataType::INT16 && out_dt == DataType::UINT16) {
        if (mode == FusionMode::ZIGZAG) { auto* s = p.addStage<DifferenceStage<int16_t, uint16_t, FusionMode::ZIGZAG>>();     s->setChunkSize(cs); return s; }
        else                            { auto* s = p.addStage<DifferenceStage<int16_t, uint16_t, FusionMode::NEGABINARY>>(); s->setChunkSize(cs); return s; }
    }
    if (in_dt == DataType::INT32 && out_dt == DataType::UINT32) {
        if (mode == FusionMode::ZIGZAG) { auto* s = p.addStage<DifferenceStage<int32_t, uint32_t, FusionMode::ZIGZAG>>();     s->setChunkSize(cs); return s; }
        else                            { auto* s = p.addStage<DifferenceStage<int32_t, uint32_t, FusionMode::NEGABINARY>>(); s->setChunkSize(cs); return s; }
    }
    if (in_dt == DataType::INT64 && out_dt == DataType::UINT64) {
        if (mode == FusionMode::ZIGZAG) { auto* s = p.addStage<DifferenceStage<int64_t, uint64_t, FusionMode::ZIGZAG>>();     s->setChunkSize(cs); return s; }
        else                            { auto* s = p.addStage<DifferenceStage<int64_t, uint64_t, FusionMode::NEGABINARY>>(); s->setChunkSize(cs); return s; }
    }

    throw std::runtime_error("loadConfig: unsupported Difference type combination");
}

static Stage* addZigzagStage(Pipeline& p, const toml::table& t) {
    DataType in_dt  = dataTypeFromString(optStr(t, "input_type",  "int32"));
    DataType out_dt = dataTypeFromString(optStr(t, "output_type", "uint32"));
    const bool bt = optBool(t, "byte_transparent", false);  // LC TCMS mode
    Stage* s = nullptr;
    if      (in_dt == DataType::INT8  && out_dt == DataType::UINT8)  { auto* z = p.addStage<ZigzagStage<int8_t,  uint8_t>>();  z->setByteTransparent(bt); s = z; }
    else if (in_dt == DataType::INT16 && out_dt == DataType::UINT16) { auto* z = p.addStage<ZigzagStage<int16_t, uint16_t>>(); z->setByteTransparent(bt); s = z; }
    else if (in_dt == DataType::INT32 && out_dt == DataType::UINT32) { auto* z = p.addStage<ZigzagStage<int32_t, uint32_t>>(); z->setByteTransparent(bt); s = z; }
    else if (in_dt == DataType::INT64 && out_dt == DataType::UINT64) { auto* z = p.addStage<ZigzagStage<int64_t, uint64_t>>(); z->setByteTransparent(bt); s = z; }
    else throw std::runtime_error("loadConfig: unsupported Zigzag type combination");
    return s;
}

static Stage* addBitpackStage(Pipeline& p, const toml::table& t) {
    DataType dt = dataTypeFromString(optStr(t, "input_type", "uint16"));
    uint8_t nbits = static_cast<uint8_t>(optInt(t, "nbits", 16));
    const bool auto_detect = optBool(t, "auto_detect", false);
    const bool auto_base   = optBool(t, "auto_base",   false);
    const bool auto_shift  = optBool(t, "auto_shift",  false);
    const bool adaptive    = optBool(t, "adaptive",    false);
    const int64_t base     = optInt(t, "base",  0);
    const uint8_t shift    = static_cast<uint8_t>(optInt(t, "shift", 0));

    auto configure = [&](auto* s) {
        s->setNBits(nbits);
        s->setBase(static_cast<decltype(s->getBase())>(base));
        s->setShift(shift);
        if (adaptive) s->setAdaptive(true);
        if (auto_detect) s->setAutoDetect(true);
        if (auto_base)   s->setAutoBase(true);
        if (auto_shift)  s->setAutoShift(true);
        return s;
    };

    if (dt == DataType::UINT8)  return configure(p.addStage<BitpackStage<uint8_t>>());
    if (dt == DataType::UINT16) return configure(p.addStage<BitpackStage<uint16_t>>());
    if (dt == DataType::UINT32) return configure(p.addStage<BitpackStage<uint32_t>>());
    throw std::runtime_error("loadConfig: unsupported Bitpack input_type");
}

static Stage* addNegabinaryStage(Pipeline& p, const toml::table& t) {
    DataType in_dt  = dataTypeFromString(optStr(t, "input_type",  "int32"));
    DataType out_dt = dataTypeFromString(optStr(t, "output_type", "uint32"));

    if (in_dt == DataType::INT8  && out_dt == DataType::UINT8)  return p.addStage<NegabinaryStage<int8_t,  uint8_t>>();
    if (in_dt == DataType::INT16 && out_dt == DataType::UINT16) return p.addStage<NegabinaryStage<int16_t, uint16_t>>();
    if (in_dt == DataType::INT32 && out_dt == DataType::UINT32) return p.addStage<NegabinaryStage<int32_t, uint32_t>>();
    if (in_dt == DataType::INT64 && out_dt == DataType::UINT64) return p.addStage<NegabinaryStage<int64_t, uint64_t>>();
    throw std::runtime_error("loadConfig: unsupported Negabinary type combination");
}

// Previously inlined in the load dispatch; now named helpers for registry use.
static Stage* addBitshuffleStage(Pipeline& p, const toml::table& t) {
    auto* bs = p.addStage<BitshuffleStage>();
    bs->setBlockSize(static_cast<size_t>(optInt(t, "block_size", 16384)));
    bs->setElementWidth(static_cast<size_t>(optInt(t, "element_width", 4)));
    return bs;
}
static Stage* addRZEStage(Pipeline& p, const toml::table& t) {
    auto* rze = p.addStage<RZEStage>();
    rze->setChunkSize(static_cast<size_t>(optInt(t, "chunk_size", 16384)));
    rze->setWordSize(static_cast<size_t>(optInt(t, "word_size", 1)));
    return rze;
}

static Stage* addRREStage(Pipeline& p, const toml::table& t) {
    auto* rre = p.addStage<RREStage>();
    rre->setChunkSize(static_cast<size_t>(optInt(t, "chunk_size", 16384)));
    rre->setWordSize(static_cast<size_t>(optInt(t, "word_size", 1)));
    return rre;
}

static Stage* addGPULZStage(Pipeline& p, const toml::table& t) {
    auto* gpulz = p.addStage<GPULZStage>();
    gpulz->setChunkSize(static_cast<size_t>(optInt(t, "chunk_size", 2048)));
    gpulz->setWordSize(static_cast<size_t>(optInt(t, "word_size", 4)));
    gpulz->setMatchLevel(static_cast<int>(optInt(t, "match_level", 1)));
    gpulz->setSplitMode(optBool(t, "split_mode", false));
    return gpulz;
}

static Stage* addRAREStage(Pipeline& p, const toml::table& t) {
    auto* rare = p.addStage<RAREStage>();
    rare->setChunkSize(static_cast<size_t>(optInt(t, "chunk_size", 16384)));
    rare->setWordSize(static_cast<size_t>(optInt(t, "word_size", 1)));
    return rare;
}

static Stage* addRAZEStage(Pipeline& p, const toml::table& t) {
    auto* raze = p.addStage<RAZEStage>();
    raze->setChunkSize(static_cast<size_t>(optInt(t, "chunk_size", 16384)));
    raze->setWordSize(static_cast<size_t>(optInt(t, "word_size", 1)));
    return raze;
}

static Stage* addCLOGStage(Pipeline& p, const toml::table& t) {
    auto* clog = p.addStage<CLOGStage>();
    clog->setChunkSize(static_cast<size_t>(optInt(t, "chunk_size", 16384)));
    clog->setWordSize(static_cast<size_t>(optInt(t, "word_size", 1)));
    return clog;
}

static Stage* addHCLOGStage(Pipeline& p, const toml::table& t) {
    auto* hclog = p.addStage<HCLOGStage>();
    hclog->setChunkSize(static_cast<size_t>(optInt(t, "chunk_size", 16384)));
    hclog->setWordSize(static_cast<size_t>(optInt(t, "word_size", 1)));
    return hclog;
}

static Stage* addTUPLStage(Pipeline& p, const toml::table& t) {
    auto* tupl = p.addStage<TUPLStage>();
    tupl->setBlockSize(static_cast<size_t>(optInt(t, "block_size", 16384)));
    tupl->setWordSize(static_cast<size_t>(optInt(t, "word_size", 1)));
    tupl->setDim(static_cast<size_t>(optInt(t, "dim", 2)));
    return tupl;
}

static Stage* addROIBinSplitStage(Pipeline& p, const toml::table& t) {
    DataType dt = dataTypeFromString(optStr(t, "data_type", "float32"));
    const auto hw   = static_cast<uint32_t>(optInt(t, "roi_half_width", 4));
    const auto bin  = static_cast<uint32_t>(optInt(t, "bin_factor", 1));
    const std::string peaks = optStr(t, "peaks_file", "");

    auto configure = [&](auto* s) {
        s->setRoiHalfWidth(hw);
        s->setBinFactor(bin);
        // setPeaksFile after the box/bin settings so the overlap statistic and
        // the geometry cross-check both see the final configuration. Dimensions
        // come from Pipeline::setDims() at addStage() time, so the peak file's
        // own geometry can be validated against them here.
        if (!peaks.empty()) s->setPeaksFile(peaks);
        return s;
    };
    if (dt == DataType::FLOAT64) return configure(p.addStage<ROIBinSplitStage<double>>());
    return configure(p.addStage<ROIBinSplitStage<float>>());
}

static Stage* addMergeStage(Pipeline& p, const toml::table& t) {
    auto* mg = p.addStage<MergeStage>();
    std::vector<std::string> names;
    if (auto* arr = t["segments"].as_array())
        for (auto& n : *arr)
            if (auto s = n.as_string()) names.push_back(s->get());
    if (names.empty())
        throw std::runtime_error("loadConfig: Merge stage requires a non-empty 'segments' array "
                                 "(one name per input, in connection order)");
    mg->setSegmentNames(names);
    return mg;
}

static Stage* addBitplaneRZEStage(Pipeline& p, const toml::table& t) {
    (void)t;  // no tunable parameters — config is derived from input length
    return p.addStage<BitplaneRZEStage>();
}

static Stage* addAdaptiveBitpackStage(Pipeline& p, const toml::table& t) {
    DataType dt = dataTypeFromString(optStr(t, "input_type", "int32"));
    uint32_t block_size = static_cast<uint32_t>(optInt(t, "block_size", 32));
    bool outlier = optBool(t, "outlier_selection", false);
    // Fused-forward-only: which warp coder policy the fused kernel composes
    // (default "AdaptiveBitpackCoder"). E.g. "PlainBitpackCoder" for a fixed-rate
    // A/B baseline. Not serialized into the archive header — the staged path and
    // the inverse are unchanged, so it never affects decode.
    std::string fused_coder = optStr(t, "fused_coder", "");
    if (dt == DataType::INT16) {
        auto* s = p.addStage<AdaptiveBitpackStage<int16_t>>();
        s->setBlockSize(block_size);
        s->setOutlierSelection(outlier);
        if (!fused_coder.empty()) s->setFusedCoder(fused_coder);
        return s;
    } else if (dt == DataType::INT32) {
        auto* s = p.addStage<AdaptiveBitpackStage<int32_t>>();
        s->setBlockSize(block_size);
        s->setOutlierSelection(outlier);
        if (!fused_coder.empty()) s->setFusedCoder(fused_coder);
        return s;
    }
    throw std::runtime_error("loadConfig: unsupported AdaptiveBitpack input_type");
}

// SZx / SZp accept float32/float64 input and error_bound / error_bound_mode
// (ABS or NOA — these whole-compressor stages have no exact per-element REL).
// Every optStr/optInt/optDbl read is kept inline in the add function body so the
// doc/TOML key checker, which parses that body, sees the full accepted-key set.
static bool szIsNoa(const std::string& m) {
    return m == "NOA" || m == "noa" || m == "REL" || m == "rel";
}

static Stage* addSZxStage(Pipeline& p, const toml::table& t) {
    DataType dt = dataTypeFromString(optStr(t, "data_type", "float32"));
    uint32_t block_size = static_cast<uint32_t>(optInt(t, "block_size", 128));
    double eb = optDbl(t, "error_bound", 1e-3);
    SZxErrorMode mode = szIsNoa(optStr(t, "error_bound_mode", "ABS"))
        ? SZxErrorMode::NOA : SZxErrorMode::ABS;
    auto configure = [&](auto* s) {
        s->setBlockSize(block_size);
        s->setErrorBound(eb);
        s->setErrorMode(mode);
        return s;
    };
    if (dt == DataType::FLOAT32) return configure(p.addStage<SZxStage<float>>());
    if (dt == DataType::FLOAT64) return configure(p.addStage<SZxStage<double>>());
    throw std::runtime_error("loadConfig: unsupported SZx data_type (use float32/float64)");
}

static Stage* addSZpStage(Pipeline& p, const toml::table& t) {
    DataType dt = dataTypeFromString(optStr(t, "data_type", "float32"));
    uint32_t block_size = static_cast<uint32_t>(optInt(t, "block_size", 128));
    double eb = optDbl(t, "error_bound", 1e-3);
    SZpErrorMode mode = szIsNoa(optStr(t, "error_bound_mode", "ABS"))
        ? SZpErrorMode::NOA : SZpErrorMode::ABS;
    auto configure = [&](auto* s) {
        s->setBlockSize(block_size);
        s->setErrorBound(eb);
        s->setErrorMode(mode);
        return s;
    };
    if (dt == DataType::FLOAT32) return configure(p.addStage<SZpStage<float>>());
    if (dt == DataType::FLOAT64) return configure(p.addStage<SZpStage<double>>());
    throw std::runtime_error("loadConfig: unsupported SZp data_type (use float32/float64)");
}

static void saveSZxStage(Stage* s, std::ostringstream& out) {
    uint8_t buf[sizeof(SZxConfig)] = {};
    size_t sz = s->serializeHeader(0, buf, sizeof(buf));
    SZxConfig cfg;
    if (sz >= sizeof(cfg)) std::memcpy(&cfg, buf, sizeof(cfg));
    out << "data_type = \""       << dataTypeToString(cfg.data_type) << "\"\n";
    out << "block_size = "        << static_cast<int64_t>(cfg.block_size) << "\n";
    out << "error_bound = "       << cfg.error_bound << "\n";
    out << "error_bound_mode = \"" << (cfg.eb_mode == 2 ? "NOA" : "ABS") << "\"\n";
}

static void saveSZpStage(Stage* s, std::ostringstream& out) {
    uint8_t buf[sizeof(SZpConfig)] = {};
    size_t sz = s->serializeHeader(0, buf, sizeof(buf));
    SZpConfig cfg;
    if (sz >= sizeof(cfg)) std::memcpy(&cfg, buf, sizeof(cfg));
    out << "data_type = \""       << dataTypeToString(cfg.data_type) << "\"\n";
    out << "block_size = "        << static_cast<int64_t>(cfg.block_size) << "\n";
    out << "error_bound = "       << cfg.error_bound << "\n";
    out << "error_bound_mode = \"" << (cfg.eb_mode == 2 ? "NOA" : "ABS") << "\"\n";
}

static Stage* addTiledLorenzoStage(Pipeline& p, const toml::table& t) {
    DataType dt = dataTypeFromString(optStr(t, "data_type", "int32"));
    uint32_t tx = static_cast<uint32_t>(optInt(t, "tile_x", 0));
    uint32_t ty = static_cast<uint32_t>(optInt(t, "tile_y", 0));
    uint32_t tz = static_cast<uint32_t>(optInt(t, "tile_z", 0));
    // Optional per-stage dimension override, for branches whose data is not the
    // pipeline's input shape (e.g. ROIBinSplit's binned background). Without it
    // finalize() would re-push the global dims over anything set here.
    const auto dx = static_cast<size_t>(optInt(t, "dim_x", 0));
    const auto dy = static_cast<size_t>(optInt(t, "dim_y", 0));
    const auto dz = static_cast<size_t>(optInt(t, "dim_z", 0));
    auto configure = [&](auto* s) {
        if (tx || ty || tz)
            s->setTileShape(tx ? tx : 1, ty ? ty : 1, tz ? tz : 1);
        if (dx) s->setDimsOverride(dx, dy ? dy : 1, dz ? dz : 1);
        return s;
    };
    if (dt == DataType::INT16) return configure(p.addStage<TiledLorenzoStage<int16_t>>());
    if (dt == DataType::INT32) return configure(p.addStage<TiledLorenzoStage<int32_t>>());
    throw std::runtime_error("loadConfig: unsupported TiledLorenzo data_type");
}

// ANSStage (vendored dietgpu) is excluded on HIP: inline NVPTX lanemask
// assembly with no translation, out of scope for the current HIP backend.
// p.addStage<ANSStage>() itself throws a
// clear runtime_error on an unsupported backend without ever referencing
// ANSStage's constructor (see Stage::isSupportedOnBackend()'s doc comment),
// so this needs no guard of its own.
static Stage* addANSStage(Pipeline& p, const toml::table& t) {
    auto* s = p.addStage<ANSStage>();
    s->setProbBits(static_cast<uint8_t>(optInt(t, "prob_bits", 10)));
    return s;
}

static Stage* addADMStage(Pipeline& p, const toml::table& t) {
    auto* s = p.addStage<ADMStage>();
    std::string dtype_str = optStr(t, "dtype", "uint16");
    if (dtype_str == "uint32") s->setDtype(ADMDtype::U32);
    return s;
}

static Stage* addHuffmanStage(Pipeline& p, const toml::table& t) {
    DataType dt = dataTypeFromString(optStr(t, "input_type", "uint16"));
    uint32_t bklen = static_cast<uint32_t>(optInt(t, "bklen", 1024));

    // Pre-built codebook.  "Adaptive" needs no parameters beyond the floor shift;
    // "Fixed" is expressible in TOML only in its model-derived form, since a raw
    // frequency table has to be supplied through the C++ API.
    const std::string book_src   = optStr(t, "book_source", "PerBlock");
    const auto        floor_shift = static_cast<uint8_t>(optInt(t, "book_floor_shift", 24));
    const auto        refit_thr   = static_cast<float>(optDbl(t, "book_refit_threshold", 1.2));
    const auto        refit_ivl   = static_cast<uint32_t>(optInt(t, "book_refit_interval", 0));
    const bool        validate_rng = optBool(t, "validate_symbol_range", true);
    const auto execution_mode = huffmanExecutionModeFromString(
        optStr(t, "execution_mode", "HostCoordinated"));
    HuffmanBookSpec spec;
    spec.model  = huffmanBookModelFromString(optStr(t, "book_model", "Gaussian"));
    spec.center = optDbl(t, "book_center", -1.0);
    spec.scale  = optDbl(t, "book_scale",  32.0);
    spec.shape  = optDbl(t, "book_shape",   2.0);

    auto configure = [&](auto* s) -> Stage* {
        s->setBklen(bklen);
        s->setAdaptiveFloorShift(floor_shift);
        s->setRefitThreshold(refit_thr);
        s->setRefitInterval(refit_ivl);
        s->setValidateSymbolRange(validate_rng);
        s->setExecutionMode(execution_mode);
        if      (book_src == "Fixed")    s->setFixedBookFromModel(spec);
        else if (book_src == "Adaptive") s->setBookSource(HuffmanBookSource::Adaptive);
        else if (book_src != "PerBlock")
            throw std::runtime_error(
                "loadConfig: unknown Huffman book_source \"" + book_src + "\"");
        return s;
    };

    if (dt == DataType::UINT8)   return configure(p.addStage<HuffmanStage<uint8_t>>());
    if (dt == DataType::UINT16)  return configure(p.addStage<HuffmanStage<uint16_t>>());
    if (dt == DataType::UINT32)  return configure(p.addStage<HuffmanStage<uint32_t>>());
    throw std::runtime_error("loadConfig: unsupported Huffman input_type \""
        + optStr(t, "input_type", "uint16") + "\"");
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage serialization helpers (save direction)
// Each saveXxxStage mirrors its addXxxStage counterpart.
// ─────────────────────────────────────────────────────────────────────────────

static void saveLorenzoStage(Stage* s, std::ostringstream& out) {
    DataType dt = static_cast<DataType>(s->getOutputDataType(0));
    out << "data_type = \"" << dataTypeToString(dt) << "\"\n";
    uint8_t buf[sizeof(LorenzoConfig)] = {};
    if (s->serializeHeader(0, buf, sizeof(buf)) >= sizeof(LorenzoConfig)) {
        LorenzoConfig lc;
        std::memcpy(&lc, buf, sizeof(lc));
        if (lc.block_size > 0)
            out << "block_size = " << static_cast<int64_t>(lc.block_size) << "\n";
    }
}

static void saveLorenzoQuantStage(Stage* s, std::ostringstream& out) {
    uint8_t buf[128] = {};
    size_t sz = s->serializeHeader(0, buf, sizeof(buf));

    DataType in_dt   = static_cast<DataType>(s->getInputDataType(0));
    DataType code_dt = static_cast<DataType>(s->getOutputDataType(0));
    if (sz >= sizeof(LorenzoQuantConfig)) {
        LorenzoQuantConfig lc;
        std::memcpy(&lc, buf, sizeof(LorenzoQuantConfig));
        in_dt   = lc.input_type;
        code_dt = lc.code_type;
    }
    out << "input_type = \"" << dataTypeToString(in_dt) << "\"\n";
    out << "code_type = \"" << dataTypeToString(code_dt) << "\"\n";

    float cap = 0.2f, eb = 1e-3f;
    ErrorBoundMode ebm = ErrorBoundMode::ABS;
    int qr = 32768;
    bool zz = false;

    auto read = [&](auto* lrz) {
        eb  = static_cast<float>(lrz->getErrorBound());
        ebm = lrz->getErrorBoundMode();
        qr  = static_cast<int>(lrz->getQuantRadius());
        cap = lrz->getOutlierCapacity();
        zz  = lrz->getZigzagCodes();
    };
    if      (in_dt == DataType::FLOAT32 && code_dt == DataType::UINT16) read(static_cast<LorenzoQuantStage<float,  uint16_t>*>(s));
    else if (in_dt == DataType::FLOAT64 && code_dt == DataType::UINT16) read(static_cast<LorenzoQuantStage<double, uint16_t>*>(s));
    else if (in_dt == DataType::FLOAT32 && code_dt == DataType::UINT8)  read(static_cast<LorenzoQuantStage<float,  uint8_t>*>(s));
    else if (in_dt == DataType::FLOAT64 && code_dt == DataType::UINT32) read(static_cast<LorenzoQuantStage<double, uint32_t>*>(s));

    out << "error_bound = "        << static_cast<double>(eb) << "\n";
    out << "error_bound_mode = \"" << ebModeToString(ebm)     << "\"\n";
    out << "quant_radius = "       << static_cast<int64_t>(qr) << "\n";
    out << "outlier_capacity = "   << static_cast<double>(cap) << "\n";
    out << "zigzag_codes = "       << (zz ? "true" : "false") << "\n";
}

static void saveQuantizerStage(Stage* s, std::ostringstream& out) {
    uint8_t buf[sizeof(QuantizerConfig)] = {};
    size_t sz = s->serializeHeader(0, buf, sizeof(buf));

    DataType in_dt   = static_cast<DataType>(s->getInputDataType(0));
    DataType code_dt = static_cast<DataType>(s->getOutputDataType(0));
    if (sz >= sizeof(QuantizerConfig)) {
        QuantizerConfig qc;
        std::memcpy(&qc, buf, sizeof(QuantizerConfig));
        in_dt   = qc.input_type;
        code_dt = qc.code_type;
    }
    out << "input_type = \"" << dataTypeToString(in_dt)   << "\"\n";
    out << "code_type = \""  << dataTypeToString(code_dt) << "\"\n";

    auto write = [&](auto* q) {
        out << "error_bound = "        << static_cast<double>(q->getErrorBound())   << "\n";
        out << "error_bound_mode = \"" << ebModeToString(q->getErrorBoundMode())    << "\"\n";
        out << "quant_radius = "       << static_cast<int64_t>(q->getQuantRadius()) << "\n";
        out << "outlier_capacity = "   << static_cast<double>(q->getOutlierCapacity()) << "\n";
        out << "zigzag_codes = "       << (q->getZigzagCodes() ? "true" : "false") << "\n";
        float thr = q->getOutlierThreshold();
        if (std::isfinite(thr)) out << "outlier_threshold = " << thr << "\n";
        if (q->getInplaceOutliers())  out << "inplace_outliers = true\n";
        if (q->getLinearMode())       out << "linear_mode = true\n";
        if (q->getLinearHighPrecision()) out << "linear_high_precision = true\n";
        if (q->getPowerOfTwoBound()) out << "power_of_two_bound = true\n";
        if (q->getDither()) {
            out << "dither = true\n";
            out << "dither_seed = " << static_cast<int64_t>(q->getDitherSeed()) << "\n";
            out << "dither_strength = " << static_cast<double>(q->getDitherStrength()) << "\n";
        }
    };
    if      (in_dt == DataType::FLOAT32 && code_dt == DataType::UINT16) write(static_cast<QuantizerStage<float,  uint16_t>*>(s));
    else if (in_dt == DataType::FLOAT32 && code_dt == DataType::UINT32) write(static_cast<QuantizerStage<float,  uint32_t>*>(s));
    else if (in_dt == DataType::FLOAT64 && code_dt == DataType::UINT16) write(static_cast<QuantizerStage<double, uint16_t>*>(s));
    else if (in_dt == DataType::FLOAT64 && code_dt == DataType::UINT32) write(static_cast<QuantizerStage<double, uint32_t>*>(s));
}

static void saveBitshuffleStage(Stage* s, std::ostringstream& out) {
    auto* bs = static_cast<BitshuffleStage*>(s);
    out << "block_size = "    << static_cast<int64_t>(bs->getBlockSize())    << "\n";
    out << "element_width = " << static_cast<int64_t>(bs->getElementWidth()) << "\n";
}

static void saveRZEStage(Stage* s, std::ostringstream& out) {
    auto* rze = static_cast<RZEStage*>(s);
    out << "chunk_size = " << static_cast<int64_t>(rze->getChunkSize()) << "\n";
    out << "word_size = "  << static_cast<int64_t>(rze->getWordSize())  << "\n";
}

static void saveRREStage(Stage* s, std::ostringstream& out) {
    auto* rre = static_cast<RREStage*>(s);
    out << "chunk_size = " << static_cast<int64_t>(rre->getChunkSize()) << "\n";
    out << "word_size = "  << static_cast<int64_t>(rre->getWordSize())  << "\n";
}

static void saveGPULZStage(Stage* s, std::ostringstream& out) {
    auto* gpulz = static_cast<GPULZStage*>(s);
    out << "chunk_size = " << static_cast<int64_t>(gpulz->getChunkSize()) << "\n";
    out << "word_size = "  << static_cast<int64_t>(gpulz->getWordSize())  << "\n";
    out << "match_level = " << static_cast<int64_t>(gpulz->getMatchLevel()) << "\n";
    out << "split_mode = " << (gpulz->getSplitMode() ? "true" : "false") << "\n";
}

static void saveRAREStage(Stage* s, std::ostringstream& out) {
    auto* rare = static_cast<RAREStage*>(s);
    out << "chunk_size = " << static_cast<int64_t>(rare->getChunkSize()) << "\n";
    out << "word_size = "  << static_cast<int64_t>(rare->getWordSize())  << "\n";
}

static void saveRAZEStage(Stage* s, std::ostringstream& out) {
    auto* raze = static_cast<RAZEStage*>(s);
    out << "chunk_size = " << static_cast<int64_t>(raze->getChunkSize()) << "\n";
    out << "word_size = "  << static_cast<int64_t>(raze->getWordSize())  << "\n";
}

static void saveCLOGStage(Stage* s, std::ostringstream& out) {
    auto* clog = static_cast<CLOGStage*>(s);
    out << "chunk_size = " << static_cast<int64_t>(clog->getChunkSize()) << "\n";
    out << "word_size = "  << static_cast<int64_t>(clog->getWordSize())  << "\n";
}

static void saveHCLOGStage(Stage* s, std::ostringstream& out) {
    auto* hclog = static_cast<HCLOGStage*>(s);
    out << "chunk_size = " << static_cast<int64_t>(hclog->getChunkSize()) << "\n";
    out << "word_size = "  << static_cast<int64_t>(hclog->getWordSize())  << "\n";
}

static Stage* addLogTransformStage(Pipeline& p, const toml::table& t) {
    auto* lg = p.addStage<LogTransformStage<float>>();
    lg->setErrorBound(static_cast<float>(optDbl(t, "error_bound", 1e-3)));
    lg->setThreshold(static_cast<float>(optDbl(t, "threshold", 0.0)));
    lg->setOutlierCapacity(static_cast<float>(optDbl(t, "outlier_capacity", 0.05)));
    return lg;
}

static void saveTUPLStage(Stage* s, std::ostringstream& out) {
    auto* tupl = static_cast<TUPLStage*>(s);
    out << "block_size = " << static_cast<int64_t>(tupl->getBlockSize()) << "\n";
    out << "word_size = "  << static_cast<int64_t>(tupl->getWordSize())  << "\n";
    out << "dim = "        << static_cast<int64_t>(tupl->getDim())       << "\n";
}

static void saveLogTransformStage(Stage* s, std::ostringstream& out) {
    auto* lg = static_cast<LogTransformStage<float>*>(s);
    out << "error_bound = "      << lg->getErrorBound()      << "\n";
    out << "threshold = "        << lg->getThreshold()       << "\n";
    out << "outlier_capacity = " << lg->getOutlierCapacity() << "\n";
}

static void saveROIBinSplitStage(Stage* s, std::ostringstream& out) {
    // The peak table travels in the archive on the `peaks` port, so a saved
    // config only needs the geometry knobs, not the path it was first read from.
    if (auto* f = dynamic_cast<ROIBinSplitStage<float>*>(s)) {
        out << "data_type = \"float32\"\n";
        out << "roi_half_width = " << f->getRoiHalfWidth() << "\n";
        out << "bin_factor = "     << f->getBinFactor()    << "\n";
    } else if (auto* d = dynamic_cast<ROIBinSplitStage<double>*>(s)) {
        out << "data_type = \"float64\"\n";
        out << "roi_half_width = " << d->getRoiHalfWidth() << "\n";
        out << "bin_factor = "     << d->getBinFactor()    << "\n";
    }
}

static Stage* addCdf97Stage(Pipeline& p, const toml::table& t) {
    // double is SPERR-bit-exact; float32 is a faster non-bit-exact variant.
    DataType dt = dataTypeFromString(optStr(t, "data_type", "float64"));
    if (dt == DataType::FLOAT32) return p.addStage<Cdf97Stage<float>>();
    return p.addStage<Cdf97Stage<double>>();
}

static void saveCdf97Stage(Stage* s, std::ostringstream& out) {
    out << "data_type = \"" << dataTypeToString(static_cast<DataType>(s->getOutputDataType(0))) << "\"\n";
}

static Stage* addSpeck2DStage(Pipeline& p, const toml::table& /*t*/) {
    return p.addStage<Speck2DStage>();
}
static void saveSpeck2DStage(Stage* /*s*/, std::ostringstream& /*out*/) {
    // No user-facing config: dims/threshold/nbitsA are pipeline/data-derived
    // and round-trip via the stage's own serializeHeader(), not the TOML file.
}

static Stage* addCdf97OutlierCorrectStage(Pipeline& p, const toml::table& t) {
    auto* s = p.addStage<Cdf97OutlierCorrectStage>();
    // MUST match the paired QuantizerStage's own error_bound (ABS mode) —
    // see the stage header's doc comment.
    s->setErrorBound(static_cast<float>(optDbl(t, "error_bound", 1e-4)));
    return s;
}
static void saveCdf97OutlierCorrectStage(Stage* s, std::ostringstream& out) {
    auto* oc = static_cast<Cdf97OutlierCorrectStage*>(s);
    out << "error_bound = " << oc->getErrorBound() << "\n";
}

static void saveMergeStage(Stage* s, std::ostringstream& out) {
    auto* mg = static_cast<MergeStage*>(s);
    const auto& names = mg->getSegmentNames();
    out << "segments = [";
    for (size_t i = 0; i < names.size(); i++) {
        out << "\"" << names[i] << "\"";
        if (i + 1 < names.size()) out << ", ";
    }
    out << "]\n";
}

static void saveBitplaneRZEStage(Stage* s, std::ostringstream& out) {
    (void)s; (void)out;  // no tunable parameters to persist
}

static void saveAdaptiveBitpackStage(Stage* s, std::ostringstream& out) {
    uint8_t buf[sizeof(AdaptiveBitpackConfig)] = {};
    size_t sz = s->serializeHeader(0, buf, sizeof(buf));
    AdaptiveBitpackConfig cfg;
    if (sz >= sizeof(cfg)) std::memcpy(&cfg, buf, sizeof(cfg));
    out << "input_type = \"" << dataTypeToString(cfg.data_type) << "\"\n";
    out << "block_size = "   << static_cast<int64_t>(cfg.block_size) << "\n";
    out << "outlier_selection = " << (cfg.outlier_selection ? "true" : "false") << "\n";
}

static void saveTiledLorenzoStage(Stage* s, std::ostringstream& out) {
    uint8_t buf[sizeof(TiledLorenzoConfig)] = {};
    size_t sz = s->serializeHeader(0, buf, sizeof(buf));
    TiledLorenzoConfig cfg;
    if (sz >= sizeof(cfg)) std::memcpy(&cfg, buf, sizeof(cfg));
    out << "data_type = \"" << dataTypeToString(cfg.data_type) << "\"\n";
    out << "tile_x = " << static_cast<int64_t>(cfg.tile_x) << "\n";
    out << "tile_y = " << static_cast<int64_t>(cfg.tile_y) << "\n";
    out << "tile_z = " << static_cast<int64_t>(cfg.tile_z) << "\n";
    // Only an explicitly pinned override is persisted; ordinary dims come from
    // Pipeline::setDims() and writing them here would freeze a saved config to
    // one input shape.
    const bool pinned =
        (dynamic_cast<TiledLorenzoStage<int32_t>*>(s) &&
         static_cast<TiledLorenzoStage<int32_t>*>(s)->hasDimsOverride()) ||
        (dynamic_cast<TiledLorenzoStage<int16_t>*>(s) &&
         static_cast<TiledLorenzoStage<int16_t>*>(s)->hasDimsOverride());
    if (pinned) {
        out << "dim_x = " << cfg.dim_x << "\n";
        out << "dim_y = " << cfg.dim_y << "\n";
        out << "dim_z = " << cfg.dim_z << "\n";
    }
}

static void saveRLEStage(Stage* s, std::ostringstream& out) {
    out << "data_type = \""
        << dataTypeToString(static_cast<DataType>(s->getOutputDataType(0))) << "\"\n";
    // chunk_size lives in the serialized stage header (after DataType and the
    // cached element count); RLEStage is a template, so it can't be down-cast
    // here without knowing T.
    uint8_t buf[16] = {};
    if (s->serializeHeader(0, buf, sizeof(buf)) >= sizeof(DataType) + 8) {
        uint32_t cs = 0;
        std::memcpy(&cs, buf + sizeof(DataType) + sizeof(uint32_t), sizeof(uint32_t));
        out << "chunk_size = " << static_cast<int64_t>(cs) << "\n";
    }
}

static void saveDifferenceStage(Stage* s, std::ostringstream& out) {
    DataType in_dt  = static_cast<DataType>(s->getInputDataType(0));
    DataType out_dt = static_cast<DataType>(s->getOutputDataType(0));
    out << "input_type = \""  << dataTypeToString(in_dt)  << "\"\n";
    out << "output_type = \"" << dataTypeToString(out_dt) << "\"\n";
    uint8_t buf[8] = {};
    size_t sz = s->serializeHeader(0, buf, sizeof(buf));
    if (sz >= 6) {
        uint32_t cs = 0;
        std::memcpy(&cs, buf + 2, sizeof(uint32_t));
        out << "chunk_size = " << static_cast<int64_t>(cs) << "\n";
    }
    if (sz >= 7 && in_dt != out_dt) {
        FusionMode mode = static_cast<FusionMode>(buf[6]);
        out << "fusion_mode = \"" << (mode == FusionMode::ZIGZAG ? "zigzag" : "negabinary") << "\"\n";
    }
}

static void saveZigzagStage(Stage* s, std::ostringstream& out) {
    // Read TIn/TOut/byte_transparent from the serialized header — getInputDataType()
    // returns UNKNOWN in byte-transparent mode, so the header is the reliable source.
    uint8_t hdr[3] = {};
    const size_t n = s->serializeHeader(0, hdr, sizeof(hdr));
    out << "input_type = \""  << dataTypeToString(static_cast<DataType>(hdr[0])) << "\"\n";
    out << "output_type = \"" << dataTypeToString(static_cast<DataType>(hdr[1])) << "\"\n";
    if (n >= 3 && hdr[2]) out << "byte_transparent = true\n";
}

static void saveNegabinaryStage(Stage* s, std::ostringstream& out) {
    out << "input_type = \""
        << dataTypeToString(static_cast<DataType>(s->getInputDataType(0)))  << "\"\n";
    out << "output_type = \""
        << dataTypeToString(static_cast<DataType>(s->getOutputDataType(0))) << "\"\n";
}

static void saveBitpackStage(Stage* s, std::ostringstream& out) {
    uint8_t buf[15] = {};
    size_t sz = s->serializeHeader(0, buf, sizeof(buf));
    DataType dt   = (sz >= 1) ? static_cast<DataType>(buf[0]) : DataType::UINT16;
    uint8_t nbits = (sz >= 2) ? buf[1] : 16;
    uint8_t shift = (sz >= 15) ? buf[10] : 0;
    uint32_t base = 0;
    if (sz >= 15) std::memcpy(&base, buf + 11, sizeof(uint32_t));
    out << "input_type = \"" << dataTypeToString(dt)       << "\"\n";
    out << "nbits = "        << static_cast<int64_t>(nbits) << "\n";
    out << "shift = "        << static_cast<int64_t>(shift) << "\n";
    out << "base = "         << static_cast<int64_t>(base)  << "\n";
}

static void saveANSStage(Stage* s, std::ostringstream& out) {
    auto* ans = static_cast<ANSStage*>(s);
    out << "prob_bits = " << static_cast<int64_t>(ans->getProbBits()) << "\n";
}

static void saveADMStage(Stage* s, std::ostringstream& out) {
    auto* adm = static_cast<ADMStage*>(s);
    out << "dtype = \"" << (adm->getDtype() == ADMDtype::U16 ? "uint16" : "uint32") << "\"\n";
}

// Add a GInterp stage (2-D or 3-D — dispatches on code_type).
// dims are NOT read from TOML — they come from `Pipeline::setDims()` which is
// already required for any multi-dim pipeline; setDims() is invoked on every
// stage at addStage time. GInterpStage::setDims throws for 1-D (dims[1] == 1).
static Stage* addGInterpStage(Pipeline& p, const toml::table& t) {
    std::string in_type   = optStr(t, "input_type", "float32");
    std::string code_type = optStr(t, "code_type", "uint16");
    DataType    in_dt     = dataTypeFromString(in_type);
    DataType    code_dt   = dataTypeFromString(code_type);

    Stage* s = nullptr;
    auto configure = [&](auto* g) {
        g->setErrorBound(static_cast<float>(optDbl(t, "error_bound", 1e-3)));
        g->setErrorBoundMode(ebModeFromString(optStr(t, "error_bound_mode", "ABS")));
        g->setQuantRadius(static_cast<int>(optInt(t, "quant_radius", 0)));
        g->setOutlierCapacity(static_cast<float>(optDbl(t, "outlier_capacity", 0.10)));
        g->setAutoTuning(static_cast<uint8_t>(optInt(t, "auto_tuning", 0)));
        s = g;
    };
    auto dispatch = [&](auto input_tag) {
        using TInput = decltype(input_tag);
        if      (code_dt == DataType::UINT16) configure(p.addStage<GInterpStage<TInput, uint16_t>>());
        else if (code_dt == DataType::UINT8)  configure(p.addStage<GInterpStage<TInput, uint8_t>>());
        else if (code_dt == DataType::UINT32) configure(p.addStage<GInterpStage<TInput, uint32_t>>());
        else
            throw std::runtime_error(
                "loadConfig: unsupported GInterp code_type \"" + code_type + "\"");
    };
    if      (in_dt == DataType::FLOAT32) dispatch(float{});
    else if (in_dt == DataType::FLOAT64) dispatch(double{});
    else
        throw std::runtime_error(
            "loadConfig: unsupported GInterp input_type \"" + in_type + "\"");
    return s;
}

static void saveGInterpStage(Stage* s, std::ostringstream& out) {
    DataType in_dt   = static_cast<DataType>(s->getInputDataType(0));
    DataType code_dt = static_cast<DataType>(s->getOutputDataType(0));
    out << "input_type = \"" << dataTypeToString(in_dt) << "\"\n";
    out << "code_type = \"" << dataTypeToString(code_dt) << "\"\n";

    float eb = 1e-3f, cap = 0.10f;
    ErrorBoundMode ebm = ErrorBoundMode::ABS;
    int qr = 0;
    uint8_t at = 0;
    auto read = [&](auto* g) {
        eb  = g->getErrorBound();
        ebm = g->getErrorBoundMode();
        qr  = g->getQuantRadius();
        cap = g->getOutlierCapacity();
        at  = g->getAutoTuningMode();
    };
    auto dispatch = [&](auto input_tag) {
        using TInput = decltype(input_tag);
        if      (code_dt == DataType::UINT16) read(static_cast<GInterpStage<TInput, uint16_t>*>(s));
        else if (code_dt == DataType::UINT8)  read(static_cast<GInterpStage<TInput, uint8_t>*>(s));
        else if (code_dt == DataType::UINT32) read(static_cast<GInterpStage<TInput, uint32_t>*>(s));
    };
    if (in_dt == DataType::FLOAT64) dispatch(double{});
    else                            dispatch(float{});
    out << "error_bound = "      << static_cast<double>(eb)  << "\n";
    out << "error_bound_mode = \"" << ebModeToString(ebm)    << "\"\n";
    out << "quant_radius = "     << static_cast<int64_t>(qr) << "\n";
    out << "outlier_capacity = " << static_cast<double>(cap) << "\n";
    // Omit auto_tuning when 0 (the default) to keep round-trip configs minimal.
    if (at != 0)
        out << "auto_tuning = " << static_cast<int>(at) << "\n";
}

static void saveHuffmanStage(Stage* s, std::ostringstream& out) {
    uint8_t buf[16] = {};
    size_t sz = s->serializeHeader(0, buf, sizeof(buf));
    DataType dt  = (sz >= 1) ? static_cast<DataType>(buf[0]) : DataType::UINT16;
    uint16_t bklen = 1024;
    if (sz >= 3) std::memcpy(&bklen, buf + 1, sizeof(uint16_t));
    out << "input_type = \"" << dataTypeToString(dt)          << "\"\n";
    out << "bklen = "        << static_cast<int64_t>(bklen)   << "\n";

    // Emit the pre-built codebook keys only for a model-derived fixed book — that is
    // the only form TOML can round-trip.  A book set from a raw frequency table saves
    // as PerBlock; the caller has to re-supply the table through the C++ API.
    auto emitBook = [&out](auto* hs) {
        if (hs->getBookSource() == HuffmanBookSource::Adaptive) {
            out << "book_source = \"Adaptive\"\n";
            if (hs->getAdaptiveFloorShift() != 24)
                out << "book_floor_shift = "
                    << static_cast<int>(hs->getAdaptiveFloorShift()) << "\n";
            if (hs->getRefitThreshold() != 1.2f)
                out << "book_refit_threshold = " << hs->getRefitThreshold() << "\n";
            if (hs->getRefitInterval() != 0)
                out << "book_refit_interval = " << hs->getRefitInterval() << "\n";
            return;
        }
        if (hs->getBookSource() != HuffmanBookSource::Fixed || !hs->hasBookSpec()) return;
        const HuffmanBookSpec& sp = hs->getBookSpec();
        out << "book_source = \"Fixed\"\n";
        out << "book_model = \""  << huffmanBookModelToString(sp.model) << "\"\n";
        out << "book_center = "   << sp.center << "\n";
        out << "book_scale = "    << sp.scale  << "\n";
        if (sp.model == HuffmanBookModel::GeneralizedNormal)
            out << "book_shape = " << sp.shape << "\n";
    };
    auto emitValidate = [&out](auto* hs) {
        if (!hs->getValidateSymbolRange()) out << "validate_symbol_range = false\n";
        if (hs->getExecutionMode() != HuffmanExecutionMode::HostCoordinated)
            out << "execution_mode = \""
                << huffmanExecutionModeToString(hs->getExecutionMode()) << "\"\n";
    };
    if      (auto* hs = dynamic_cast<HuffmanStage<uint8_t>*>(s))  { emitBook(hs); emitValidate(hs); }
    else if (auto* hs = dynamic_cast<HuffmanStage<uint16_t>*>(s)) { emitBook(hs); emitValidate(hs); }
    else if (auto* hs = dynamic_cast<HuffmanStage<uint32_t>*>(s)) { emitBook(hs); emitValidate(hs); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage registry
//
// Single location for all load + save dispatch.
// To add a new stage type:
//   1. #include its header at the top of this file
//   2. Add addXxxStage / saveXxxStage helpers above
//   3. Append one entry to kStageRegistry below, INCLUDING its source_dir
//
// `source_dir` is the stage's module directory, relative to the repo root. It is
// consumed by scripts/gen_stage_fingerprints.py, which hashes each stage's sources
// plus their transitive local includes so downstream consumers can tell whether a
// stage's code changed between two builds (see include/pipeline/config.h,
// fz::stageFingerprints()). The generator FAILS the build if an entry names a
// directory that does not exist, or if a stage is missing one — so this stays
// honest without anyone having to remember it.
// ─────────────────────────────────────────────────────────────────────────────

struct StageEntry {
    const char*  type_name;  // TOML "type" string (load and save)
    StageType    enum_val;   // matches getStageTypeId() for the save direction
    Stage*       (*load_fn)(Pipeline&, const toml::table&);
    void         (*save_fn)(Stage*, std::ostringstream&);
    const char*  source_dir; // module dir, repo-relative; see note above
};

static const StageEntry kStageRegistry[] = {
    { "Lorenzo",      StageType::LORENZO,      addLorenzoStage,      saveLorenzoStage,      "modules/predictors/lorenzo" },
    { "LorenzoQuant", StageType::LORENZO_QUANT, addLorenzoQuantStage, saveLorenzoQuantStage, "modules/fused/lorenzo_quant" },
    { "AdaptiveLorenzo", StageType::ADAPTIVE_LORENZO, addAdaptiveLorenzoStage, saveAdaptiveLorenzoStage, "modules/fused/adaptive_lorenzo" },
    { "Quantizer",    StageType::QUANTIZER,    addQuantizerStage,    saveQuantizerStage,    "modules/quantizers/quantizer" },
    { "Bitshuffle",   StageType::BITSHUFFLE,   addBitshuffleStage,   saveBitshuffleStage,   "modules/shufflers/bitshuffle" },
    { "RZE",          StageType::RZE,          addRZEStage,          saveRZEStage,          "modules/coders/rze" },
    { "RRE",          StageType::RRE,          addRREStage,          saveRREStage,          "modules/coders/rre" },
    { "GPULZ",        StageType::GPULZ,        addGPULZStage,        saveGPULZStage,        "modules/coders/gpulz" },
    { "RARE",         StageType::RARE,         addRAREStage,         saveRAREStage,         "modules/coders/rare" },
    { "RAZE",         StageType::RAZE,         addRAZEStage,         saveRAZEStage,         "modules/coders/raze" },
    { "CLOG",         StageType::CLOG,         addCLOGStage,         saveCLOGStage,         "modules/coders/clog" },
    { "HCLOG",        StageType::HCLOG,        addHCLOGStage,        saveHCLOGStage,        "modules/coders/hclog" },
    { "TUPL",         StageType::TUPL,         addTUPLStage,         saveTUPLStage,         "modules/shufflers/tupl" },
    { "LogTransform", StageType::LOG_TRANSFORM, addLogTransformStage, saveLogTransformStage, "modules/transforms/log_transform" },
    { "CDF97",        StageType::CDF97,         addCdf97Stage,        saveCdf97Stage,        "modules/transforms/cdf97" },
    { "SPECK2D",      StageType::SPECK2D,       addSpeck2DStage,      saveSpeck2DStage,      "modules/coders/speck2d" },
    { "Cdf97OutlierCorrect", StageType::CDF97_OUTLIER_CORRECT, addCdf97OutlierCorrectStage, saveCdf97OutlierCorrectStage, "modules/coders/cdf97_outlier_correct" },
    { "Merge",        StageType::MERGE,        addMergeStage,        saveMergeStage,        "modules/structural/merge" },
    { "ROIBinSplit",  StageType::ROIBIN_SPLIT, addROIBinSplitStage,  saveROIBinSplitStage,  "modules/structural/roibin_split" },
    { "RLE",          StageType::RLE,          addRLEStage,          saveRLEStage,          "modules/coders/rle" },
    { "Difference",   StageType::DIFFERENCE,   addDifferenceStage,   saveDifferenceStage,   "modules/predictors/diff" },
    { "Zigzag",       StageType::ZIGZAG,       addZigzagStage,       saveZigzagStage,       "modules/transforms/zigzag" },
    { "Negabinary",   StageType::NEGABINARY,   addNegabinaryStage,   saveNegabinaryStage,   "modules/transforms/negabinary" },
    { "Bitpack",      StageType::BITPACK,      addBitpackStage,      saveBitpackStage,      "modules/coders/bitpack" },
    { "Huffman",      StageType::HUFFMAN,      addHuffmanStage,      saveHuffmanStage,      "modules/coders/huffman" },
    { "ANS",          StageType::ANS,          addANSStage,          saveANSStage,          "modules/coders/ans" },
    { "ADM",          StageType::ADM,          addADMStage,          saveADMStage,          "modules/transforms/adm" },
    { "GInterp",      StageType::G_INTERP,     addGInterpStage,      saveGInterpStage,      "modules/fused/ginterp" },
    { "BitplaneRZE",  StageType::BITPLANE_RZE, addBitplaneRZEStage,  saveBitplaneRZEStage,  "modules/fused/bitplane_rze" },
    { "AdaptiveBitpack", StageType::ADAPTIVE_BITPACK, addAdaptiveBitpackStage, saveAdaptiveBitpackStage, "modules/coders/adaptive_bitpack" },
    { "TiledLorenzo", StageType::TILED_LORENZO, addTiledLorenzoStage, saveTiledLorenzoStage, "modules/predictors/tiled_lorenzo" },
    { "SZx",          StageType::SZX,          addSZxStage,          saveSZxStage,          "modules/fused/szx" },
    { "SZp",          StageType::SZP,          addSZpStage,          saveSZpStage,          "modules/fused/szp" },
};

// Declared in include/pipeline/config.h.  Deliberately derived from the registry
// above rather than maintained separately — see the header for why.
std::vector<std::string> registeredStageTypes() {
    std::vector<std::string> out;
    out.reserve(sizeof(kStageRegistry) / sizeof(kStageRegistry[0]));
    for (const auto& e : kStageRegistry) out.emplace_back(e.type_name);
    return out;
}

std::vector<StageFingerprintInfo> stageFingerprints() {
    std::vector<StageFingerprintInfo> out;
    out.reserve(sizeof(kStageRegistry) / sizeof(kStageRegistry[0]));
    for (const auto& e : kStageRegistry) {
        StageFingerprintInfo info{e.type_name, ""};
        // Linear scan over ~25 entries, once: not worth a map.  A stage with no
        // generated entry reports an empty fingerprint rather than a fabricated
        // one, so a consumer can tell "unchanged" from "unknown".
        for (const auto& g : generated::kStageFingerprints) {
            if (info.name == g.name) { info.fingerprint = g.fingerprint; break; }
        }
        out.push_back(std::move(info));
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline::loadConfig()
// ─────────────────────────────────────────────────────────────────────────────

void Pipeline::loadConfig(const std::string& path) {
    if (is_finalized_) {
        throw std::runtime_error("loadConfig: pipeline is already finalized");
    }

    // ── Parse TOML ───────────────────────────────────────────────────────────
    toml::table doc;
    try {
        doc = toml::parse_file(path);
    } catch (const toml::parse_error& e) {
        throw std::runtime_error(
            std::string("loadConfig: failed to parse \"") + path + "\": " + e.what());
    }

    // ── Pipeline-level settings ───────────────────────────────────────────────
    // Resolved against stage_map (built below) right before finalize() --
    // see Pipeline::setPrimarySource()'s doc comment for what this controls.
    std::string primary_source_name;
    if (auto* pl = doc["pipeline"].as_table()) {
        if (auto v = (*pl)["primary_source"].as_string())
            primary_source_name = v->get();

        if (auto v = (*pl)["input_size"].as_integer())
            input_size_hint_ = static_cast<size_t>(v->get());

        if (auto* da = (*pl)["dims"].as_array()) {
            std::array<size_t, 3> d = {0, 1, 1};
            for (size_t i = 0; i < 3 && i < da->size(); ++i)
                if (auto iv = (*da)[i].as_integer())
                    d[i] = static_cast<size_t>(iv->get());
            setDims(d);
        }

        if (auto v = (*pl)["memory_strategy"].as_string())
            setMemoryStrategy(strategyFromString(v->get()));

        if (auto v = (*pl)["pool_multiplier"].as_floating_point())
            pool_multiplier_ = static_cast<float>(v->get());
        else if (auto vi = (*pl)["pool_multiplier"].as_integer())
            pool_multiplier_ = static_cast<float>(vi->get());

        if (auto v = (*pl)["num_streams"].as_integer())
            setNumStreams(static_cast<int>(v->get()));

        // Buffer coloring.  Read *after* memory_strategy on purpose: setMemoryStrategy()
        // rebuilds the DAG and re-applies the stored coloring flag, so setting coloring
        // first would be re-applied rather than overwritten — but relying on that
        // ordering silently is how this setting got lost once already (a plain
        // setColoringEnabled(false) before loadConfig() was discarded, making the
        // uncolored arm of the peak-memory sweep a no-op).  Keep it last and explicit.
        if (auto v = (*pl)["coloring"].as_boolean())
            setColoringEnabled(v->get());
    }

    // ── Stage construction pass ───────────────────────────────────────────────
    // Maps config-local stage name → Stage* for wiring below.
    std::unordered_map<std::string, Stage*> stage_map;
    // Preserve stage order for the wiring pass.
    struct StageEntry { std::string name; Stage* ptr; const toml::table* tbl; };
    std::vector<StageEntry> entries;

    auto* stage_arr = doc["stage"].as_array();
    if (!stage_arr) {
        throw std::runtime_error("loadConfig: no [[stage]] entries found in \"" + path + "\"");
    }

    for (auto& node : *stage_arr) {
        auto* t = node.as_table();
        if (!t) continue;

        auto name_node = (*t)["name"].as_string();
        auto type_node = (*t)["type"].as_string();
        if (!name_node || !type_node)
            throw std::runtime_error("loadConfig: each [[stage]] must have 'name' and 'type'");

        std::string name = name_node->get();
        std::string type = type_node->get();

        if (stage_map.count(name))
            throw std::runtime_error("loadConfig: duplicate stage name \"" + name + "\"");

        Stage* s = nullptr;
        for (const auto& entry : kStageRegistry) {
            if (type == entry.type_name) { s = entry.load_fn(*this, *t); break; }
        }
        if (!s)
            throw std::runtime_error("loadConfig: unknown stage type \"" + type + "\"");

        stage_map[name] = s;
        entries.push_back({name, s, t});
    }

    // ── Wiring pass ───────────────────────────────────────────────────────────
    for (auto& entry : entries) {
        auto* inp_arr = (*entry.tbl)["inputs"].as_array();
        if (!inp_arr || inp_arr->empty()) continue;  // source stage

        for (auto& inp_node : *inp_arr) {
            auto* inp = inp_node.as_table();
            if (!inp) continue;

            auto from_node = (*inp)["from"].as_string();
            if (!from_node)
                throw std::runtime_error(
                    "loadConfig: stage \"" + entry.name + "\" input missing 'from' key");

            std::string from = from_node->get();

            // Reserved sentinel: this input port binds directly to the
            // pipeline's external buffer (Pipeline::bindExternalInput()),
            // not to another stage's output. Position in the `inputs` array
            // is significant, same as for a real connection -- list it in
            // the port order the stage expects.
            if (from == "__external__") {
                bindExternalInput(entry.ptr);
                continue;
            }

            auto it = stage_map.find(from);
            if (it == stage_map.end())
                throw std::runtime_error(
                    "loadConfig: stage \"" + entry.name
                    + "\" references unknown stage \"" + from + "\"");

            std::string port = "output";
            if (auto pn = (*inp)["port"].as_string()) port = pn->get();

            connect(entry.ptr, it->second, port);
        }
    }

    // ── Primary source (multi-source pipelines only) ───────────────────────────
    if (!primary_source_name.empty()) {
        auto it = stage_map.find(primary_source_name);
        if (it == stage_map.end())
            throw std::runtime_error(
                "loadConfig: primary_source references unknown stage \"" + primary_source_name + "\"");
        setPrimarySource(it->second);
    }

    // ── Finalize ──────────────────────────────────────────────────────────────
    finalize();
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline::saveConfig()
// ─────────────────────────────────────────────────────────────────────────────

void Pipeline::saveConfig(const std::string& path) const {
    if (!is_finalized_) {
        throw std::runtime_error("saveConfig: pipeline must be finalized first");
    }

    // Build a reverse map: Stage* → local name (use getName() + disambig suffix)
    std::unordered_map<Stage*, std::string> stage_names;
    std::unordered_map<std::string, int> name_counts;
    for (auto& s : stages_) {
        std::string base = s->getName();
        int cnt = name_counts[base]++;
        stage_names[s.get()] = (cnt == 0) ? base : base + std::to_string(cnt);
    }

    std::ostringstream out;
    out << "# FZGPUModules pipeline config\n"
        << "# Generated by Pipeline::saveConfig(). Load with Pipeline::loadConfig().\n\n";

    out << "[pipeline]\n";
    out << "input_size = " << static_cast<int64_t>(input_size_hint_) << "\n";
    out << "memory_strategy = \"" << strategyToString(strategy_) << "\"\n";
    // Only emitted when non-default, so round-tripping a normal pipeline does not
    // grow a key nobody set. Round-trips through loadConfig()'s "coloring" reader.
    if (!coloring_enabled_) out << "coloring = false\n";
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "pool_multiplier = " << static_cast<double>(pool_multiplier_) << "\n";
    out << "num_streams = " << static_cast<int64_t>(num_streams_) << "\n";
    out << "dims = [" << static_cast<int64_t>(dims_[0]) << ", "
        << static_cast<int64_t>(dims_[1]) << ", "
        << static_cast<int64_t>(dims_[2]) << "]\n";
    if (primary_source_stage_)
        out << "primary_source = \"" << tomlEscape(stage_names.at(primary_source_stage_)) << "\"\n";
    out << "\n";

    for (auto& s_uptr : stages_) {
        Stage* s = s_uptr.get();
        std::string local_name = stage_names.at(s);
        uint16_t type_id = s->getStageTypeId();
        StageType stype  = static_cast<StageType>(type_id);

        const StageEntry* entry = nullptr;
        for (const auto& e : kStageRegistry) {
            if (e.enum_val == stype) { entry = &e; break; }
        }

        out << "[[stage]]\n";
        out << "name = \"" << tomlEscape(local_name) << "\"\n";
        out << "type = \"" << (entry ? entry->type_name : stageTypeToString(stype)) << "\"\n";
        if (entry) entry->save_fn(s, out);

        // inputs: collect real connections AND bindExternalInput() bindings
        // that have this stage as dependent, merged in actual port order.
        // Both connect() and bindExternalInput() append to the same
        // DAGNode::input_buffer_ids in call order, so that vector's index is
        // the authoritative port position -- look each one up rather than
        // assuming real connections always precede external bindings or
        // vice versa (Cdf97OutlierCorrectStage happens to bind external
        // first today, but nothing enforces that in general).
        DAGNode* s_node = stage_to_node_.at(s);
        std::vector<std::string> port_entries(s_node->input_buffer_ids.size());
        for (auto& conn : connections_) {
            if (conn.dependent != s) continue;
            DAGNode* prod_node = stage_to_node_.at(conn.producer);
            int buf_id = prod_node->output_index_to_buffer_id.at(conn.output_index);
            auto it = std::find(s_node->input_buffer_ids.begin(), s_node->input_buffer_ids.end(), buf_id);
            if (it == s_node->input_buffer_ids.end())
                throw std::runtime_error("saveConfig: internal error -- connection buffer not found on dependent node");
            size_t pos = static_cast<size_t>(it - s_node->input_buffer_ids.begin());
            std::ostringstream e;
            e << "{ from = \"" << tomlEscape(stage_names.at(conn.producer)) << "\"";
            if (conn.output_name != "output")
                e << ", port = \"" << tomlEscape(conn.output_name) << "\"";
            e << " }";
            port_entries[pos] = e.str();
        }
        for (const auto& [ext_node, ext_buf_id] : explicit_external_bindings_) {
            if (ext_node != s_node) continue;
            auto it = std::find(s_node->input_buffer_ids.begin(), s_node->input_buffer_ids.end(), ext_buf_id);
            if (it == s_node->input_buffer_ids.end())
                throw std::runtime_error("saveConfig: internal error -- external binding buffer not found on its own node");
            size_t pos = static_cast<size_t>(it - s_node->input_buffer_ids.begin());
            port_entries[pos] = "{ from = \"__external__\" }";
        }
        bool has_inputs = !port_entries.empty() && std::any_of(
            port_entries.begin(), port_entries.end(), [](const std::string& e) { return !e.empty(); });
        if (has_inputs) {
            out << "inputs = [";
            for (size_t i = 0; i < port_entries.size(); i++) {
                if (i > 0) out << ", ";
                out << port_entries[i];
            }
            out << "]\n";
        }

        out << "\n";
    }

    // ── Write to file ─────────────────────────────────────────────────────────
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("saveConfig: cannot open \"" + path + "\" for writing");
    f << out.str();
    if (!f)
        throw std::runtime_error("saveConfig: write error on \"" + path + "\"");
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline(config_path) constructor
// ─────────────────────────────────────────────────────────────────────────────

Pipeline::Pipeline(const std::string& config_path)
    : Pipeline()  // delegate to default constructor
{
    loadConfig(config_path);
}

} // namespace fz
