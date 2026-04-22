/**
 * @file config.cpp
 * @brief TOML config file load/save for Pipeline.
 *
 * toml++ is only included here — it never leaks into public headers.
 */

// toml++ in header-only mode
#define TOML_HEADER_ONLY 1
#include <toml++/toml.hpp>

#include "pipeline/compressor.h"

// All stage types supported by loadConfig / saveConfig
#include "predictors/lorenzo/lorenzo.h"
#include "predictors/quantizer/quantizer.h"
#include "transforms/bitshuffle/bitshuffle_stage.h"
#include "transforms/rze/rze_stage.h"
#include "transforms/zigzag/zigzag_stage.h"
#include "transforms/negabinary/negabinary_stage.h"
#include "encoders/RLE/rle.h"
#include "encoders/diff/diff.h"

#include <fstream>
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

// Add a Lorenzo stage (dispatches on input_type / code_type strings).
static Stage* addLorenzoStage(Pipeline& p, const toml::table& t) {
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
        configure(p.addStage<LorenzoStage<float, uint16_t>>());
    else if (in_dt == DataType::FLOAT64 && code_dt == DataType::UINT16)
        configure(p.addStage<LorenzoStage<double, uint16_t>>());
    else if (in_dt == DataType::FLOAT32 && code_dt == DataType::UINT8)
        configure(p.addStage<LorenzoStage<float, uint8_t>>());
    else if (in_dt == DataType::FLOAT64 && code_dt == DataType::UINT32)
        configure(p.addStage<LorenzoStage<double, uint32_t>>());
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

static Stage* addNegabinaryStage(Pipeline& p, const toml::table& t) {
    DataType in_dt  = dataTypeFromString(optStr(t, "input_type",  "int32"));
    DataType out_dt = dataTypeFromString(optStr(t, "output_type", "uint32"));

    // Import NegabinaryStage — lives in the negabinary module header
    using namespace fz;
    if (in_dt == DataType::INT8  && out_dt == DataType::UINT8)  return p.addStage<NegabinaryStage<int8_t,  uint8_t>>();
    if (in_dt == DataType::INT16 && out_dt == DataType::UINT16) return p.addStage<NegabinaryStage<int16_t, uint16_t>>();
    if (in_dt == DataType::INT32 && out_dt == DataType::UINT32) return p.addStage<NegabinaryStage<int32_t, uint32_t>>();
    if (in_dt == DataType::INT64 && out_dt == DataType::UINT64) return p.addStage<NegabinaryStage<int64_t, uint64_t>>();
    throw std::runtime_error("loadConfig: unsupported Negabinary type combination");
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

        if (type == "Lorenzo1D" || type == "Lorenzo2D" || type == "Lorenzo3D") {
            s = addLorenzoStage(*this, *t);
        } else if (type == "Quantizer") {
            s = addQuantizerStage(*this, *t);
        } else if (type == "Bitshuffle") {
            auto* bs = addStage<BitshuffleStage>();
            bs->setBlockSize(static_cast<size_t>(optInt(*t, "block_size", 16384)));
            bs->setElementWidth(static_cast<size_t>(optInt(*t, "element_width", 4)));
            s = bs;
        } else if (type == "RZE") {
            auto* rze = addStage<RZEStage>();
            rze->setChunkSize(static_cast<size_t>(optInt(*t, "chunk_size", 16384)));
            rze->setLevels(static_cast<int>(optInt(*t, "levels", 4)));
            s = rze;
        } else if (type == "RLE") {
            s = addRLEStage(*this, *t);
        } else if (type == "Difference") {
            s = addDifferenceStage(*this, *t);
        } else if (type == "Zigzag") {
            s = addZigzagStage(*this, *t);
        } else if (type == "Negabinary") {
            s = addNegabinaryStage(*this, *t);
        } else {
            throw std::runtime_error("loadConfig: unknown stage type \"" + type + "\"");
        }

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

    // ── Build TOML document ───────────────────────────────────────────────────
    toml::table doc;

    // [pipeline]
    toml::table pl_tbl;
    pl_tbl.insert("input_size",      static_cast<int64_t>(input_size_hint_));
    pl_tbl.insert("memory_strategy", strategyToString(strategy_));
    pl_tbl.insert("pool_multiplier", static_cast<double>(pool_multiplier_));
    pl_tbl.insert("num_streams",     static_cast<int64_t>(num_streams_));
    toml::array dims_arr;
    dims_arr.push_back(static_cast<int64_t>(dims_[0]));
    dims_arr.push_back(static_cast<int64_t>(dims_[1]));
    dims_arr.push_back(static_cast<int64_t>(dims_[2]));
    pl_tbl.insert("dims", std::move(dims_arr));
    doc.insert("pipeline", std::move(pl_tbl));

    // [[stage]] array — one table per stage in construction order
    toml::array stages_arr;

    for (auto& s_uptr : stages_) {
        Stage* s = s_uptr.get();
        toml::table st;

        std::string local_name = stage_names.at(s);
        st.insert("name", local_name);

        uint16_t type_id = s->getStageTypeId();
        StageType stype  = static_cast<StageType>(type_id);
        st.insert("type", stageTypeToString(stype));

        // Per-type parameters — write human-readable keys
        switch (stype) {
            case StageType::LORENZO_1D:
            case StageType::LORENZO_2D:
            case StageType::LORENZO_3D: {
                // Use serializeHeader to read back the LorenzoConfig struct,
                // which gives us the canonical type IDs. Then use public getters
                // for the user-facing parameters (error_bound, eb_mode, etc.),
                // since those are always valid post-finalize regardless of whether
                // compress() has been called.
                uint8_t buf[128] = {};
                size_t sz = s->serializeHeader(0, buf, sizeof(buf));

                // Determine type strings via the binary config (most reliable source)
                // or fall back to the output/input DataType methods.
                DataType in_dt   = static_cast<DataType>(s->getInputDataType(0));
                DataType code_dt = static_cast<DataType>(s->getOutputDataType(0)); // codes port

                if (sz >= sizeof(LorenzoConfig)) {
                    LorenzoConfig lc;
                    std::memcpy(&lc, buf, sizeof(LorenzoConfig));
                    in_dt   = lc.input_type;
                    code_dt = lc.code_type;
                }

                st.insert("input_type", dataTypeToString(in_dt));
                st.insert("code_type",  dataTypeToString(code_dt));

                // All remaining params come from public getters — always valid.
                float cap = 0.2f;
                float eb  = 1e-3f;
                ErrorBoundMode ebm = ErrorBoundMode::ABS;
                int   qr  = 32768;
                bool  zz  = false;

                if (in_dt == DataType::FLOAT32 && code_dt == DataType::UINT16) {
                    auto* lrz = static_cast<LorenzoStage<float, uint16_t>*>(s);
                    eb = static_cast<float>(lrz->getErrorBound()); ebm = lrz->getErrorBoundMode();
                    qr = static_cast<int>(lrz->getQuantRadius()); cap = lrz->getOutlierCapacity();
                    zz = lrz->getZigzagCodes();
                } else if (in_dt == DataType::FLOAT64 && code_dt == DataType::UINT16) {
                    auto* lrz = static_cast<LorenzoStage<double, uint16_t>*>(s);
                    eb = static_cast<float>(lrz->getErrorBound()); ebm = lrz->getErrorBoundMode();
                    qr = static_cast<int>(lrz->getQuantRadius()); cap = lrz->getOutlierCapacity();
                    zz = lrz->getZigzagCodes();
                } else if (in_dt == DataType::FLOAT32 && code_dt == DataType::UINT8) {
                    auto* lrz = static_cast<LorenzoStage<float, uint8_t>*>(s);
                    eb = static_cast<float>(lrz->getErrorBound()); ebm = lrz->getErrorBoundMode();
                    qr = static_cast<int>(lrz->getQuantRadius()); cap = lrz->getOutlierCapacity();
                    zz = lrz->getZigzagCodes();
                } else if (in_dt == DataType::FLOAT64 && code_dt == DataType::UINT32) {
                    auto* lrz = static_cast<LorenzoStage<double, uint32_t>*>(s);
                    eb = static_cast<float>(lrz->getErrorBound()); ebm = lrz->getErrorBoundMode();
                    qr = static_cast<int>(lrz->getQuantRadius()); cap = lrz->getOutlierCapacity();
                    zz = lrz->getZigzagCodes();
                }

                st.insert("error_bound",       static_cast<double>(eb));
                st.insert("error_bound_mode",  ebModeToString(ebm));
                st.insert("quant_radius",      static_cast<int64_t>(qr));
                st.insert("outlier_capacity",  static_cast<double>(cap));
                st.insert("zigzag_codes",      zz);
                break;
            }
            case StageType::BITSHUFFLE: {
                auto* bs = static_cast<BitshuffleStage*>(s);
                st.insert("block_size",    static_cast<int64_t>(bs->getBlockSize()));
                st.insert("element_width", static_cast<int64_t>(bs->getElementWidth()));
                break;
            }
            case StageType::RZE: {
                auto* rze = static_cast<RZEStage*>(s);
                st.insert("chunk_size", static_cast<int64_t>(rze->getChunkSize()));
                st.insert("levels",     static_cast<int64_t>(rze->getLevels()));
                break;
            }
            case StageType::RLE: {
                // DataType is baked into the template; read it from the stage output type
                st.insert("data_type",
                    dataTypeToString(static_cast<DataType>(s->getOutputDataType(0))));
                break;
            }
            case StageType::DIFFERENCE: {
                st.insert("input_type",
                    dataTypeToString(static_cast<DataType>(s->getInputDataType(0))));
                st.insert("output_type",
                    dataTypeToString(static_cast<DataType>(s->getOutputDataType(0))));
                // chunk_size: stored in serialized header bytes 2-5
                uint8_t buf[8] = {};
                size_t sz = s->serializeHeader(0, buf, sizeof(buf));
                if (sz >= 6) {
                    uint32_t cs = 0;
                    std::memcpy(&cs, buf + 2, sizeof(uint32_t));
                    st.insert("chunk_size", static_cast<int64_t>(cs));
                }
                break;
            }
            case StageType::ZIGZAG:
            case StageType::NEGABINARY: {
                st.insert("input_type",
                    dataTypeToString(static_cast<DataType>(s->getInputDataType(0))));
                st.insert("output_type",
                    dataTypeToString(static_cast<DataType>(s->getOutputDataType(0))));
                break;
            }
            default:
                break;  // Unknown/unsupported — emit type string only
        }

        // inputs: collect connections that have this stage as dependent
        toml::array inp_arr;
        for (auto& conn : connections_) {
            if (conn.dependent != s) continue;
            toml::table inp_tbl;
            inp_tbl.insert("from", stage_names.at(conn.producer));
            if (conn.output_name != "output")
                inp_tbl.insert("port", conn.output_name);
            inp_arr.push_back(std::move(inp_tbl));
        }
        if (!inp_arr.empty())
            st.insert("inputs", std::move(inp_arr));

        stages_arr.push_back(std::move(st));
    }

    doc.insert("stage", std::move(stages_arr));

    // ── Write to file ─────────────────────────────────────────────────────────
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("saveConfig: cannot open \"" + path + "\" for writing");
    f << "# FZGPUModules pipeline config\n"
      << "# Generated by Pipeline::saveConfig(). Load with Pipeline::loadConfig().\n\n"
      << doc << "\n";
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
