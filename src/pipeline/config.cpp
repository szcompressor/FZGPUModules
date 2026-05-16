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

// All stage types supported by loadConfig / saveConfig
#include "fused/lorenzo_quant/lorenzo_quant.h"
#include "predictors/lorenzo/lorenzo_stage.h"
#include "quantizers/quantizer/quantizer.h"
#include "shufflers/bitshuffle/bitshuffle_stage.h"
#include "coders/rze/rze_stage.h"
#include "transforms/zigzag/zigzag_stage.h"
#include "transforms/negabinary/negabinary_stage.h"
#include "coders/bitpack/bitpack_stage.h"
#include "coders/huffman/huffman_stage.h"
#include "coders/rle/rle.h"
#include "predictors/diff/diff.h"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

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
    if (s == "ABS") return ErrorBoundMode::ABS;
    if (s == "REL") return ErrorBoundMode::REL;
    if (s == "NOA") return ErrorBoundMode::NOA;
    throw std::runtime_error("loadConfig: unknown error_bound_mode \"" + s + "\"");
}

static std::string ebModeToString(ErrorBoundMode m) {
    switch (m) {
        case ErrorBoundMode::REL: return "REL";
        case ErrorBoundMode::NOA: return "NOA";
        default:                  return "ABS";
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

    Stage* s = nullptr;
    auto configure = [&](auto* lrz) {
        s = lrz;
    };

    if      (dt == DataType::INT8)  configure(p.addStage<LorenzoStage<int8_t>>());
    else if (dt == DataType::INT16) configure(p.addStage<LorenzoStage<int16_t>>());
    else if (dt == DataType::INT32) configure(p.addStage<LorenzoStage<int32_t>>());
    else if (dt == DataType::INT64) configure(p.addStage<LorenzoStage<int64_t>>());
    else
        throw std::runtime_error(
            "loadConfig: unsupported Lorenzo data_type \"" + dt_str + "\"");

    return s;
}

// Add a LorenzoQuant stage (dispatches on input_type / code_type strings).
static Stage* addLorenzoQuantStage(Pipeline& p, const toml::table& t) {
    std::string in_type   = optStr(t, "input_type", "float32");
    std::string code_type = optStr(t, "code_type",  "uint16");

    DataType in_dt   = dataTypeFromString(in_type);
    DataType code_dt = dataTypeFromString(code_type);

    Stage* s = nullptr;

    auto configure = [&](auto* lrz) {
        lrz->setErrorBound(static_cast<float>(optDbl(t, "error_bound", 1e-3)));
        lrz->setErrorBoundMode(ebModeFromString(optStr(t, "error_bound_mode", "ABS")));
        lrz->setQuantRadius(static_cast<int>(optInt(t, "quant_radius", 32768)));
        lrz->setOutlierCapacity(static_cast<float>(optDbl(t, "outlier_capacity", 0.2)));
        lrz->setZigzagCodes(optBool(t, "zigzag_codes", false));
        s = lrz;
    };

    if (in_dt == DataType::FLOAT32 && code_dt == DataType::UINT16)
        configure(p.addStage<LorenzoQuantStage<float, uint16_t>>());
    else if (in_dt == DataType::FLOAT64 && code_dt == DataType::UINT16)
        configure(p.addStage<LorenzoQuantStage<double, uint16_t>>());
    else if (in_dt == DataType::FLOAT32 && code_dt == DataType::UINT8)
        configure(p.addStage<LorenzoQuantStage<float, uint8_t>>());
    else if (in_dt == DataType::FLOAT64 && code_dt == DataType::UINT32)
        configure(p.addStage<LorenzoQuantStage<double, uint32_t>>());
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

    auto configure = [&](auto* quant) {
        quant->setErrorBound(static_cast<float>(optDbl(t, "error_bound", 1e-3)));
        quant->setErrorBoundMode(ebModeFromString(optStr(t, "error_bound_mode", "REL")));
        quant->setQuantRadius(static_cast<int>(optInt(t, "quant_radius", 32768)));
        quant->setOutlierCapacity(static_cast<float>(optDbl(t, "outlier_capacity", 0.05)));
        quant->setZigzagCodes(optBool(t, "zigzag_codes", true));
        
        float threshold = static_cast<float>(optDbl(t, "outlier_threshold", std::numeric_limits<float>::infinity()));
        if (std::isfinite(threshold)) {
            quant->setOutlierThreshold(threshold);
        }
        
        quant->setInplaceOutliers(optBool(t, "inplace_outliers", false));
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
    switch (dt) {
        case DataType::UINT8:  return p.addStage<RLEStage<uint8_t>>();
        case DataType::UINT16: return p.addStage<RLEStage<uint16_t>>();
        case DataType::UINT32: return p.addStage<RLEStage<uint32_t>>();
        case DataType::INT32:  return p.addStage<RLEStage<int32_t>>();
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
    // Negabinary-fused instantiations (signed → unsigned of same width)
    size_t cs = static_cast<size_t>(optInt(t, "chunk_size", 0));
    if (in_dt == DataType::INT8  && out_dt == DataType::UINT8)  { auto* s = p.addStage<DifferenceStage<int8_t,  uint8_t>>();  s->setChunkSize(cs); return s; }
    if (in_dt == DataType::INT16 && out_dt == DataType::UINT16) { auto* s = p.addStage<DifferenceStage<int16_t, uint16_t>>(); s->setChunkSize(cs); return s; }
    if (in_dt == DataType::INT32 && out_dt == DataType::UINT32) { auto* s = p.addStage<DifferenceStage<int32_t, uint32_t>>(); s->setChunkSize(cs); return s; }
    if (in_dt == DataType::INT64 && out_dt == DataType::UINT64) { auto* s = p.addStage<DifferenceStage<int64_t, uint64_t>>(); s->setChunkSize(cs); return s; }

    throw std::runtime_error("loadConfig: unsupported Difference type combination");
}

static Stage* addZigzagStage(Pipeline& p, const toml::table& t) {
    DataType in_dt  = dataTypeFromString(optStr(t, "input_type",  "int32"));
    DataType out_dt = dataTypeFromString(optStr(t, "output_type", "uint32"));
    if (in_dt == DataType::INT8  && out_dt == DataType::UINT8)  return p.addStage<ZigzagStage<int8_t,  uint8_t>>();
    if (in_dt == DataType::INT16 && out_dt == DataType::UINT16) return p.addStage<ZigzagStage<int16_t, uint16_t>>();
    if (in_dt == DataType::INT32 && out_dt == DataType::UINT32) return p.addStage<ZigzagStage<int32_t, uint32_t>>();
    if (in_dt == DataType::INT64 && out_dt == DataType::UINT64) return p.addStage<ZigzagStage<int64_t, uint64_t>>();
    throw std::runtime_error("loadConfig: unsupported Zigzag type combination");
}

static Stage* addBitpackStage(Pipeline& p, const toml::table& t) {
    DataType dt = dataTypeFromString(optStr(t, "input_type", "uint16"));
    uint8_t nbits = static_cast<uint8_t>(optInt(t, "nbits", 16));
    if (dt == DataType::UINT8) {
        auto* s = p.addStage<BitpackStage<uint8_t>>();
        s->setNBits(nbits);
        return s;
    } else if (dt == DataType::UINT16) {
        auto* s = p.addStage<BitpackStage<uint16_t>>();
        s->setNBits(nbits);
        return s;
    } else if (dt == DataType::UINT32) {
        auto* s = p.addStage<BitpackStage<uint32_t>>();
        s->setNBits(nbits);
        return s;
    }
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
    rze->setLevels(static_cast<int>(optInt(t, "levels", 4)));
    return rze;
}

static Stage* addHuffmanStage(Pipeline& p, const toml::table& t) {
    DataType dt = dataTypeFromString(optStr(t, "input_type", "uint16"));
    uint32_t bklen = static_cast<uint32_t>(optInt(t, "bklen", 1024));
    if (dt == DataType::UINT8) {
        auto* s = p.addStage<HuffmanStage<uint8_t>>();
        s->setBklen(bklen);
        return s;
    } else if (dt == DataType::UINT16) {
        auto* s = p.addStage<HuffmanStage<uint16_t>>();
        s->setBklen(bklen);
        return s;
    } else if (dt == DataType::UINT32) {
        auto* s = p.addStage<HuffmanStage<uint32_t>>();
        s->setBklen(bklen);
        return s;
    }
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
    out << "levels = "     << static_cast<int64_t>(rze->getLevels())    << "\n";
}

static void saveRLEStage(Stage* s, std::ostringstream& out) {
    out << "data_type = \""
        << dataTypeToString(static_cast<DataType>(s->getOutputDataType(0))) << "\"\n";
}

static void saveDifferenceStage(Stage* s, std::ostringstream& out) {
    out << "input_type = \""
        << dataTypeToString(static_cast<DataType>(s->getInputDataType(0)))  << "\"\n";
    out << "output_type = \""
        << dataTypeToString(static_cast<DataType>(s->getOutputDataType(0))) << "\"\n";
    uint8_t buf[8] = {};
    size_t sz = s->serializeHeader(0, buf, sizeof(buf));
    if (sz >= 6) {
        uint32_t cs = 0;
        std::memcpy(&cs, buf + 2, sizeof(uint32_t));
        out << "chunk_size = " << static_cast<int64_t>(cs) << "\n";
    }
}

static void saveZigzagStage(Stage* s, std::ostringstream& out) {
    out << "input_type = \""
        << dataTypeToString(static_cast<DataType>(s->getInputDataType(0)))  << "\"\n";
    out << "output_type = \""
        << dataTypeToString(static_cast<DataType>(s->getOutputDataType(0))) << "\"\n";
}

static void saveNegabinaryStage(Stage* s, std::ostringstream& out) {
    out << "input_type = \""
        << dataTypeToString(static_cast<DataType>(s->getInputDataType(0)))  << "\"\n";
    out << "output_type = \""
        << dataTypeToString(static_cast<DataType>(s->getOutputDataType(0))) << "\"\n";
}

static void saveBitpackStage(Stage* s, std::ostringstream& out) {
    uint8_t buf[10] = {};
    size_t sz = s->serializeHeader(0, buf, sizeof(buf));
    DataType dt   = (sz >= 1) ? static_cast<DataType>(buf[0]) : DataType::UINT16;
    uint8_t nbits = (sz >= 2) ? buf[1] : 16;
    out << "input_type = \"" << dataTypeToString(dt)       << "\"\n";
    out << "nbits = "        << static_cast<int64_t>(nbits) << "\n";
}

static void saveHuffmanStage(Stage* s, std::ostringstream& out) {
    uint8_t buf[16] = {};
    size_t sz = s->serializeHeader(0, buf, sizeof(buf));
    DataType dt  = (sz >= 1) ? static_cast<DataType>(buf[0]) : DataType::UINT16;
    uint16_t bklen = 1024;
    if (sz >= 3) std::memcpy(&bklen, buf + 1, sizeof(uint16_t));
    out << "input_type = \"" << dataTypeToString(dt)          << "\"\n";
    out << "bklen = "        << static_cast<int64_t>(bklen)   << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage registry
//
// Single location for all load + save dispatch.
// To add a new stage type:
//   1. #include its header at the top of this file
//   2. Add addXxxStage / saveXxxStage helpers above
//   3. Append one entry to kStageRegistry below
// ─────────────────────────────────────────────────────────────────────────────

struct StageEntry {
    const char*  type_name;  // TOML "type" string (load and save)
    StageType    enum_val;   // matches getStageTypeId() for the save direction
    Stage*       (*load_fn)(Pipeline&, const toml::table&);
    void         (*save_fn)(Stage*, std::ostringstream&);
};

static const StageEntry kStageRegistry[] = {
    { "Lorenzo",      StageType::LORENZO,      addLorenzoStage,      saveLorenzoStage      },
    { "LorenzoQuant", StageType::LORENZO_QUANT, addLorenzoQuantStage, saveLorenzoQuantStage },
    { "Quantizer",    StageType::QUANTIZER,    addQuantizerStage,    saveQuantizerStage    },
    { "Bitshuffle",   StageType::BITSHUFFLE,   addBitshuffleStage,   saveBitshuffleStage   },
    { "RZE",          StageType::RZE,          addRZEStage,          saveRZEStage          },
    { "RLE",          StageType::RLE,          addRLEStage,          saveRLEStage          },
    { "Difference",   StageType::DIFFERENCE,   addDifferenceStage,   saveDifferenceStage   },
    { "Zigzag",       StageType::ZIGZAG,       addZigzagStage,       saveZigzagStage       },
    { "Negabinary",   StageType::NEGABINARY,   addNegabinaryStage,   saveNegabinaryStage   },
    { "Bitpack",      StageType::BITPACK,      addBitpackStage,      saveBitpackStage      },
    { "Huffman",      StageType::HUFFMAN,      addHuffmanStage,      saveHuffmanStage      },
};

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
    if (auto* pl = doc["pipeline"].as_table()) {
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
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "pool_multiplier = " << static_cast<double>(pool_multiplier_) << "\n";
    out << "num_streams = " << static_cast<int64_t>(num_streams_) << "\n";
    out << "dims = [" << static_cast<int64_t>(dims_[0]) << ", "
        << static_cast<int64_t>(dims_[1]) << ", "
        << static_cast<int64_t>(dims_[2]) << "]\n\n";

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

        // inputs: collect connections that have this stage as dependent
        bool has_inputs = false;
        std::ostringstream inputs;
        for (auto& conn : connections_) {
            if (conn.dependent != s) continue;
            if (!has_inputs) {
                inputs << "inputs = [";
                has_inputs = true;
            } else {
                inputs << ", ";
            }
            inputs << "{ from = \"" << tomlEscape(stage_names.at(conn.producer)) << "\"";
            if (conn.output_name != "output")
                inputs << ", port = \"" << tomlEscape(conn.output_name) << "\"";
            inputs << " }";
        }
        if (has_inputs) {
            inputs << "]\n";
            out << inputs.str();
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
