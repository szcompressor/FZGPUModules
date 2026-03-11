#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace fz {

// ===== FZM File Format Constants =====
// Following cuSZ's pattern: fixed-size header with offset table

constexpr uint32_t FZM_MAGIC = 0x464D5A32;  // "FZM2" in hex

// Version encoding: high byte = major, low byte = minor.
// Major mismatch (reader.major != file.major) → throw.
// Minor mismatch (reader.minor != file.minor) → warn, continue.
// Old files written before the split used a bare integer (e.g. 3); those are
// treated as major = value, minor = 0, so FZM_VERSION = 0x0300 is backward-
// compatible with files that stored version = 3.
constexpr uint8_t  FZM_VERSION_MAJOR = 3;
constexpr uint8_t  FZM_VERSION_MINOR = 0;
constexpr uint16_t FZM_VERSION = (static_cast<uint16_t>(FZM_VERSION_MAJOR) << 8)
                                | static_cast<uint16_t>(FZM_VERSION_MINOR);

// Extract major/minor from a raw version field read from disk.
// Files written before the major/minor split stored a small integer (e.g. 3);
// values ≤ 0xFF are treated as (major = value, minor = 0).
constexpr uint8_t fzmVersionMajor(uint16_t v) {
    return (v <= 0xFF) ? static_cast<uint8_t>(v) : static_cast<uint8_t>(v >> 8);
}
constexpr uint8_t fzmVersionMinor(uint16_t v) {
    return (v <= 0xFF) ? 0u : static_cast<uint8_t>(v & 0xFF);
}

constexpr size_t FZM_MAX_BUFFERS = 32;     // Max pipeline outputs
constexpr size_t FZM_MAX_NAME_LEN = 64;    // Max output name length
constexpr size_t FZM_STAGE_CONFIG_SIZE = 128;  // Max stage config size
constexpr size_t FZM_MAX_SOURCES = 4;      // Max pipeline source stages

// ===== Stage Type IDs =====
// Used to identify which stage produced a buffer

enum class StageType : uint16_t {
    UNKNOWN = 0,
    LORENZO_1D = 1,
    DIFFERENCE = 2,
    SCALE = 3,
    PASSTHROUGH = 4,
    RLE = 5,
    HUFFMAN = 6,
    BITPACK = 7,
    SPLIT = 10,
    MERGE = 11,
    LORENZO_2D = 12,
    LORENZO_3D = 13,
    QUANTIZER = 14,
    ZIGZAG     = 15,
    NEGABINARY = 16,
    BITSHUFFLE = 17,
    RZE        = 18,
};

// ===== Data Type IDs =====
// Used to identify data types in buffers

enum class DataType : uint8_t {
    UINT8 = 0,
    UINT16 = 1,
    UINT32 = 2,
    UINT64 = 3,
    INT8 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    FLOAT32 = 8,
    FLOAT64 = 9,
};

// ===== Stage Info =====
// Describes a stage in the pipeline DAG
// Used for reconstructing execution order during decompression

constexpr size_t FZM_MAX_STAGE_INPUTS = 8;
constexpr size_t FZM_MAX_STAGE_OUTPUTS = 8;

struct FZMStageInfo {
    StageType stage_type;                // Type of stage (2B)
    uint16_t stage_version;              // Config format version (2B)
    uint8_t num_inputs;                  // Number of input buffers (1B)
    uint8_t num_outputs;                 // Number of output buffers (1B)
    uint16_t reserved1;                  // Padding (2B)
    
    // Buffer connections (which buffers this stage uses/produces)
    uint16_t input_buffer_ids[FZM_MAX_STAGE_INPUTS];   // Input buffer indices (16B)
    uint16_t output_buffer_ids[FZM_MAX_STAGE_OUTPUTS]; // Output buffer indices (16B)
    
    // Stage configuration data
    uint8_t stage_config[FZM_STAGE_CONFIG_SIZE];  // Stage-specific config (128B)
    uint32_t config_size;                         // Actual bytes used (4B)
    
    uint8_t reserved2[84];  // Future expansion (84B)
    
    // Total: 256 bytes (cache-line aligned)
    // Breakdown: 2+2+1+1+2+16+16+128+4+84 = 256
    
    FZMStageInfo() {
        stage_type = StageType::UNKNOWN;
        stage_version = 0;
        num_inputs = 0;
        num_outputs = 0;
        reserved1 = 0;
        memset(input_buffer_ids, 0xFF, sizeof(input_buffer_ids));  // -1 = unused
        memset(output_buffer_ids, 0xFF, sizeof(output_buffer_ids));
        memset(stage_config, 0, FZM_STAGE_CONFIG_SIZE);
        config_size = 0;
        memset(reserved2, 0, 84);
    }
};
static_assert(sizeof(FZMStageInfo) == 256, "FZMStageInfo must be 256 bytes");

// ===== Buffer Entry =====
// Describes one buffer segment in the data payload

struct FZMBufferEntry {
    StageType stage_type;        // Which stage produced this (2B)
    uint16_t stage_version;      // Producer stage config version (2B)
    DataType data_type;          // Type of data in buffer (1B)
    uint8_t producer_output_idx; // Which output port (1B)
    uint16_t dag_buffer_id;      // DAG buffer ID for inverse routing (2B)
    char name[FZM_MAX_NAME_LEN]; // Output name (64B) - null terminated
    
    // Size information
    uint64_t data_size;          // Actual compressed data size (8B)
    uint64_t allocated_size;     // Buffer capacity for decompression (8B)
    uint64_t uncompressed_size;  // Size after decompressing this stage (8B)
    
    // File layout (offsets into data buffer after header)
    uint64_t byte_offset;        // Where this buffer starts (8B)
    
    // Stage-specific config (optional, can use FZMStageInfo instead)
    uint8_t stage_config[FZM_STAGE_CONFIG_SIZE];   // Stage-specific data (128B)
    uint32_t config_size;        // Actual bytes used in stage_config (4B)
    
    uint8_t reserved2[14];       // Future expansion (14B)
    
    // Total: 256 bytes per entry (cache-line friendly)
    
    FZMBufferEntry() {
        stage_type = StageType::UNKNOWN;
        stage_version = 0;
        data_type = DataType::UINT8;
        producer_output_idx = 0;
        dag_buffer_id = 0xFFFF;  // Sentinel: not yet assigned
        memset(name, 0, FZM_MAX_NAME_LEN);
        data_size = 0;
        allocated_size = 0;
        uncompressed_size = 0;
        byte_offset = 0;
        memset(stage_config, 0, FZM_STAGE_CONFIG_SIZE);
        config_size = 0;
        memset(reserved2, 0, 14);
    }
};
static_assert(sizeof(FZMBufferEntry) == 256, "FZMBufferEntry must be 256 bytes");

// ===== Main Header =====
// Variable-length format: FZMHeaderCore followed by FZMStageInfo[] and FZMBufferEntry[]
//
// On-disk layout:
//   [FZMHeaderCore]                           (72 bytes)
//   [FZMStageInfo × num_stages]               (256 × num_stages bytes)
//   [FZMBufferEntry × num_buffers]            (256 × num_buffers bytes)
//   [compressed data payload]                 (compressed_size bytes)
//
// header_size tells where the data begins
// For a 2-stage, 3-buffer pipeline: 72 + 512 + 768 = 1352 bytes

struct FZMHeaderCore {
    // Magic and version
    uint32_t magic;              // FZM_MAGIC (4B)
    uint16_t version;            // FZM_VERSION (2B)
    uint16_t num_buffers;        // Number of buffer segments (2B)
    
    // Global metadata
    uint64_t uncompressed_size;  // Sum of all source sizes (8B) — kept for pool sizing
    uint64_t compressed_size;    // Total compressed data size (8B)
    uint64_t header_size;        // Total header size including stage/buffer arrays (8B)
    
    // Pipeline metadata
    uint32_t num_stages;         // Number of stages in pipeline (4B)
    uint16_t num_sources;        // Number of source (input) stages (2B)
    uint16_t reserved1;          // Padding (2B)
    
    // Per-source uncompressed sizes (v3+).
    // Indexed in forward topological source-discovery order (same order
    // as the InputSpec vector passed to compress()).
    // source_uncompressed_sizes[0..num_sources-1] are valid.
    // For single-source pipelines num_sources==1 and
    // source_uncompressed_sizes[0] == uncompressed_size.
    uint64_t source_uncompressed_sizes[FZM_MAX_SOURCES];  // (32B)
    
    // Total: 4+2+2+8+8+8+4+2+2+32 = 72 bytes
    
    FZMHeaderCore() {
        magic = FZM_MAGIC;
        version = FZM_VERSION;
        num_buffers = 0;
        uncompressed_size = 0;
        compressed_size = 0;
        header_size = sizeof(FZMHeaderCore);
        num_stages = 0;
        num_sources = 0;
        reserved1 = 0;
        memset(source_uncompressed_sizes, 0, sizeof(source_uncompressed_sizes));
    }
    
    /** Compute total header size from stage/buffer counts */
    uint64_t computeHeaderSize() const {
        return sizeof(FZMHeaderCore)
             + num_stages * sizeof(FZMStageInfo)
             + num_buffers * sizeof(FZMBufferEntry);
    }
};
static_assert(sizeof(FZMHeaderCore) == 72, "FZMHeaderCore must be 72 bytes");

// ===== Stage-Specific Config Structures =====
// 
// NOTE: Stage-specific configs are NOT defined here!
// Each stage defines its own config structure in its header file.
//
// The FZMBufferEntry.stage_config[128] is a generic buffer that
// stages serialize into via Stage::serializeHeader().
//
// Examples:
//   - modules/predictors/lorenzo/lorenzo.h defines LorenzoConfig
//   - modules/encoders/huffman/huffman.h defines HuffmanConfig
//   - modules/transforms/difference.h defines DifferenceConfig
//
// This keeps fzm_format.h generic and independent of stage implementations.
// The DAG/Pipeline just calls stage->serializeHeader() and doesn't need
// to know the specific config structure.
//
// Stage config guidelines:
//   - Must fit in FZM_STAGE_CONFIG_SIZE (128 bytes)
//   - Should be POD (plain old data) for memcpy safety
//   - Include version/reserved fields for future compatibility
//   - Document byte layout in stage header file

// ===== Helper Functions =====

/**
 * Get size of data type in bytes
 */
inline size_t getDataTypeSize(DataType type) {
    switch (type) {
        case DataType::UINT8:
        case DataType::INT8:
            return 1;
        case DataType::UINT16:
        case DataType::INT16:
            return 2;
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::FLOAT32:
            return 4;
        case DataType::UINT64:
        case DataType::INT64:
        case DataType::FLOAT64:
            return 8;
        default:
            throw std::runtime_error("Unknown data type");
    }
}

/**
 * Convert DataType to string for debugging
 */
inline std::string dataTypeToString(DataType type) {
    switch (type) {
        case DataType::UINT8: return "uint8";
        case DataType::UINT16: return "uint16";
        case DataType::UINT32: return "uint32";
        case DataType::UINT64: return "uint64";
        case DataType::INT8: return "int8";
        case DataType::INT16: return "int16";
        case DataType::INT32: return "int32";
        case DataType::INT64: return "int64";
        case DataType::FLOAT32: return "float32";
        case DataType::FLOAT64: return "float64";
        default: return "unknown";
    }
}

/**
 * Convert StageType to string for debugging
 */
inline std::string stageTypeToString(StageType type) {
    switch (type) {
        case StageType::LORENZO_1D: return "Lorenzo1D";
        case StageType::LORENZO_2D: return "Lorenzo2D";
        case StageType::LORENZO_3D: return "Lorenzo3D";
        case StageType::DIFFERENCE: return "Difference";
        case StageType::SCALE: return "Scale";
        case StageType::PASSTHROUGH: return "PassThrough";
        case StageType::RLE: return "RLE";
        case StageType::HUFFMAN: return "Huffman";
        case StageType::BITPACK: return "BitPack";
        case StageType::SPLIT: return "Split";
        case StageType::MERGE: return "Merge";
        case StageType::QUANTIZER:  return "Quantizer";
        case StageType::ZIGZAG:     return "Zigzag";
        case StageType::NEGABINARY: return "Negabinary";
        case StageType::BITSHUFFLE: return "Bitshuffle";
        case StageType::RZE:        return "RZE";
        default: return "Unknown";
    }
}

} // namespace fz