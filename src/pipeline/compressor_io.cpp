// compressor_io.cpp — file-based serialization: buildHeader, writeToFile,
// readHeader, loadCompressedData, decompressFromFile, getOutputBuffers.
#include "pipeline/compressor.h"
#include "fzm_format.h"
#include "log.h"
#include "cuda_check.h"
#include <array>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <unordered_set>

namespace fz {

// ─── CRC32 (IEEE 802.3 / zlib polynomial) ─────────────────────────────────
// Used to detect silent data corruption in FZM files.

namespace {

// Lazy-initialized CRC32 lookup table (computed once on first call).
static const std::array<uint32_t, 256>& crc32Table() {
    static const auto t = []() {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++)
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    return t;
}

// Feed bytes into a running CRC register (pass the previous register value as
// `crc`; initialize with 0xFFFFFFFF and XOR result with 0xFFFFFFFF to finalize).
static uint32_t crc32Feed(uint32_t crc, const void* data, size_t len) {
    const auto& table = crc32Table();
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; i++)
        crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    return crc;
}

// Compute CRC32 of a single contiguous region.
static uint32_t crc32(const void* data, size_t len) {
    return crc32Feed(0xFFFFFFFFu, data, len) ^ 0xFFFFFFFFu;
}

// Compute CRC32 over the entire FZM header (core + stage array + buffer array),
// with header_checksum zeroed so the field doesn't contribute to its own hash.
static uint32_t computeHeaderChecksum(const Pipeline::FZMFileHeader& fh) {
    FZMHeaderCore core_zero = fh.core;
    core_zero.header_checksum = 0;
    uint32_t crc = 0xFFFFFFFFu;
    crc = crc32Feed(crc, &core_zero, sizeof(FZMHeaderCore));
    if (!fh.stages.empty())
        crc = crc32Feed(crc, fh.stages.data(),
                        fh.stages.size() * sizeof(FZMStageInfo));
    if (!fh.buffers.empty())
        crc = crc32Feed(crc, fh.buffers.data(),
                        fh.buffers.size() * sizeof(FZMBufferEntry));
    return crc ^ 0xFFFFFFFFu;
}

} // anonymous namespace

// ── Local helper: build per-level timing summary (mirrors compressor_exec.cpp) ──
static std::vector<LevelTimingResult> buildLevelTimings(
    const std::vector<StageTimingResult>& stages
) {
    std::unordered_map<int, LevelTimingResult> level_map;
    for (const auto& st : stages) {
        auto& lv = level_map[st.level];
        lv.level = st.level;
        lv.parallelism++;
        lv.elapsed_ms = std::max(lv.elapsed_ms, st.elapsed_ms);
    }
    std::vector<LevelTimingResult> levels;
    for (auto& [lvl, lv] : level_map) levels.push_back(lv);
    std::sort(levels.begin(), levels.end(),
              [](const LevelTimingResult& a, const LevelTimingResult& b) {
                  return a.level < b.level;
              });
    return levels;
}

// ========== File Serialization ==========

std::vector<Pipeline::OutputBuffer> Pipeline::getOutputBuffers() const {
    std::vector<OutputBuffer> buffers;

    for (const auto& meta : buffer_metadata_) {
        OutputBuffer buf;
        buf.d_ptr          = dag_->getBuffer(meta.buffer_id);
        buf.actual_size    = meta.actual_size;
        buf.allocated_size = meta.allocated_size;
        buf.name           = meta.name;
        buf.buffer_id      = meta.buffer_id;
        buffers.push_back(buf);
    }

    return buffers;
}

Pipeline::FZMFileHeader Pipeline::buildHeader() const {
    if (!is_finalized_) {
        throw std::runtime_error("Cannot build header before finalization");
    }
    if (buffer_metadata_.empty()) {
        throw std::runtime_error("Cannot build header before compress() is called");
    }

    FZMFileHeader fh;
    fh.core.uncompressed_size = input_size_;

    // Per-source uncompressed sizes (v3+).
    // Recorded by compress() in input_nodes_ order (= source discovery order).
    fh.core.num_sources = static_cast<uint16_t>(
        std::min(input_nodes_.size(), FZM_MAX_SOURCES));
    for (uint16_t i = 0; i < fh.core.num_sources; i++) {
        size_t sz = (i < source_input_sizes_.size()) ? source_input_sizes_[i] : 0;
        fh.core.source_uncompressed_sizes[i] = static_cast<uint64_t>(sz);
    }

    // Build stage information from DAG topology (in execution order)
    const auto& levels = dag_->getLevels();

    for (const auto& level : levels) {
        for (auto* node : level) {
            FZMStageInfo stage_info;
            stage_info.stage_type    = static_cast<StageType>(node->stage->getStageTypeId());
            stage_info.stage_version = 1;
            stage_info.num_inputs    = static_cast<uint8_t>(node->input_buffer_ids.size());
            stage_info.num_outputs   = static_cast<uint8_t>(node->output_buffer_ids.size());

            for (size_t i = 0; i < node->input_buffer_ids.size() && i < FZM_MAX_STAGE_INPUTS; i++) {
                stage_info.input_buffer_ids[i] = static_cast<uint16_t>(node->input_buffer_ids[i]);
            }
            for (size_t i = 0; i < node->output_buffer_ids.size() && i < FZM_MAX_STAGE_OUTPUTS; i++) {
                stage_info.output_buffer_ids[i] = static_cast<uint16_t>(node->output_buffer_ids[i]);
            }

            if (stage_info.num_outputs > 0) {
                stage_info.config_size = static_cast<uint32_t>(
                    node->stage->serializeHeader(0, stage_info.stage_config, FZM_STAGE_CONFIG_SIZE)
                );
            }
            fh.stages.push_back(stage_info);
        }
    }
    fh.core.num_stages = static_cast<uint32_t>(fh.stages.size());

    // Build buffer entries
    fh.core.num_buffers = static_cast<uint16_t>(buffer_metadata_.size());
    uint64_t byte_offset = 0;

    for (uint16_t i = 0; i < buffer_metadata_.size(); i++) {
        const auto& meta = buffer_metadata_[i];
        FZMBufferEntry entry;

        entry.stage_type   = static_cast<StageType>(meta.producer->stage->getStageTypeId());
        entry.stage_version = 1;
        entry.data_type    = static_cast<DataType>(
            meta.producer->stage->getOutputDataType(meta.output_index)
        );
        entry.producer_output_idx = static_cast<uint8_t>(meta.output_index);
        entry.dag_buffer_id       = static_cast<uint16_t>(meta.buffer_id);  // for inverse routing
        strncpy(entry.name, meta.name.c_str(), FZM_MAX_NAME_LEN - 1);
        entry.name[FZM_MAX_NAME_LEN - 1] = '\0';
        entry.data_size        = meta.actual_size;
        entry.allocated_size   = meta.allocated_size;
        entry.uncompressed_size = meta.actual_size;
        entry.byte_offset      = byte_offset;
        entry.config_size      = static_cast<uint32_t>(
            meta.producer->stage->serializeHeader(meta.output_index, entry.stage_config, FZM_STAGE_CONFIG_SIZE)
        );

        byte_offset += meta.actual_size;
        fh.buffers.push_back(entry);
    }

    fh.core.compressed_size = byte_offset;
    fh.core.header_size     = fh.core.computeHeaderSize();

    FZ_LOG(INFO, "Built FZM header: %u stages, %u buffers, %u source(s), %.2f MB compressed, header %llu bytes",
           fh.core.num_stages, fh.core.num_buffers, fh.core.num_sources,
           fh.core.compressed_size / (1024.0 * 1024.0),
           (unsigned long long)fh.core.header_size);

    return fh;
}

void Pipeline::writeToFile(const std::string& filename, cudaStream_t stream) {
    if (!is_finalized_) {
        throw std::runtime_error("Cannot write to file before finalization");
    }
    if (!is_compressed_) {
        throw std::runtime_error(
            "compress() must be called before writeToFile()");
    }

    FZMFileHeader fh = buildHeader();

    size_t total_data_size = fh.core.compressed_size;
    void* h_data = malloc(total_data_size);
    if (!h_data) {
        throw std::runtime_error(
            "Failed to allocate host buffer for file write (" +
            std::to_string(total_data_size) + " bytes)");
    }

    size_t offset = 0;
    for (const auto& meta : buffer_metadata_) {
        void* d_buffer = dag_->getBuffer(meta.buffer_id);
        cudaError_t err = cudaMemcpyAsync(
            static_cast<uint8_t*>(h_data) + offset,
            d_buffer, meta.actual_size,
            cudaMemcpyDeviceToHost, stream
        );
        if (err != cudaSuccess) {
            free(h_data);
            throw std::runtime_error("cudaMemcpyAsync failed: " + std::string(cudaGetErrorString(err)));
        }
        offset += meta.actual_size;
    }

    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        free(h_data);
        throw std::runtime_error("cudaStreamSynchronize failed: " + std::string(cudaGetErrorString(err)));
    }

    // Compute data checksum over the compressed payload (host copy).
    fh.core.data_checksum = crc32(h_data, total_data_size);

    // Set all feature flags before computing the header checksum so the flags
    // field has its final on-disk value when it is hashed.  header_checksum
    // is still 0 at this point, which is what computeHeaderChecksum() expects.
    fh.core.flags |= FZM_FLAG_HAS_DATA_CHECKSUM | FZM_FLAG_HAS_HEADER_CHECKSUM;

    // Compute header checksum over FZMHeaderCore + stage array + buffer array
    // (with header_checksum zeroed — see computeHeaderChecksum()).
    fh.core.header_checksum = computeHeaderChecksum(fh);

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        free(h_data);
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Write header core + stage array + buffer array
    file.write(reinterpret_cast<const char*>(&fh.core), sizeof(FZMHeaderCore));
    if (!fh.stages.empty()) {
        file.write(reinterpret_cast<const char*>(fh.stages.data()),
                   fh.stages.size() * sizeof(FZMStageInfo));
    }
    if (!fh.buffers.empty()) {
        file.write(reinterpret_cast<const char*>(fh.buffers.data()),
                   fh.buffers.size() * sizeof(FZMBufferEntry));
    }
    if (!file) {
        free(h_data);
        throw std::runtime_error("Failed to write header to file: " + filename);
    }

    file.write(reinterpret_cast<const char*>(h_data), total_data_size);
    if (!file) {
        free(h_data);
        throw std::runtime_error("Failed to write compressed data to file: " + filename);
    }

    file.close();
    free(h_data);

    size_t total_file_size = fh.core.header_size + total_data_size;
    FZ_LOG(INFO, "Wrote %.2f MB to %s (Header: %llu bytes, Data: %.2f MB)",
           total_file_size / (1024.0 * 1024.0), filename.c_str(),
           (unsigned long long)fh.core.header_size,
           total_data_size / (1024.0 * 1024.0));
}

Pipeline::FZMFileHeader Pipeline::readHeader(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    // ── Step 1: probe magic + version before committing to a read size ────────
    uint32_t probe_magic   = 0;
    uint16_t probe_version = 0;
    file.read(reinterpret_cast<char*>(&probe_magic),   sizeof(probe_magic));
    file.read(reinterpret_cast<char*>(&probe_version), sizeof(probe_version));
    if (!file) {
        throw std::runtime_error("Failed to read header from file: " + filename);
    }

    if (probe_magic != FZM_MAGIC) {
        throw std::runtime_error("Invalid FZM file format (bad magic number)");
    }

    uint8_t file_major = fzmVersionMajor(probe_version);
    uint8_t file_minor = fzmVersionMinor(probe_version);

    if (file_major != FZM_VERSION_MAJOR) {
        throw std::runtime_error(
            "Incompatible FZM major version: file has major " +
            std::to_string(file_major) +
            ", reader expects major " +
            std::to_string(FZM_VERSION_MAJOR)
        );
    }
    const bool version_exact = (file_minor == FZM_VERSION_MINOR);
    if (!version_exact) {
        FZ_LOG(WARN,
               "FZM minor version mismatch: file=%u.%u, reader=%u.%u — "
               "proceeding (forward/backward minor compat)",
               file_major, file_minor,
               FZM_VERSION_MAJOR, FZM_VERSION_MINOR);
    }

    // ── Step 2: seek back and read the core header ────────────────────────────
    // v3.0 files had a 72-byte core; v3.1+ is 80 bytes.  Reading exactly the
    // right number of bytes keeps the file cursor aligned for the stage array.
    file.seekg(0, std::ios::beg);
    FZMFileHeader fh;
    fh.core = FZMHeaderCore{};  // zero-init so new fields default to 0 for legacy files

    const bool is_legacy = (file_minor == 0);  // v3.0 had no checksum fields
    const size_t core_read_size = is_legacy ? FZM_LEGACY_HEADER_CORE_SIZE
                                            : sizeof(FZMHeaderCore);
    file.read(reinterpret_cast<char*>(&fh.core), core_read_size);
    if (!file) {
        throw std::runtime_error("Failed to read header from file: " + filename);
    }

    if (fh.core.num_stages > FZM_MAX_BUFFERS || fh.core.num_buffers > FZM_MAX_BUFFERS) {
        throw std::runtime_error("FZM header has too many stages/buffers");
    }

    // ── Step 3: read stage and buffer arrays ──────────────────────────────────
    fh.stages.resize(fh.core.num_stages);
    if (fh.core.num_stages > 0) {
        file.read(reinterpret_cast<char*>(fh.stages.data()),
                  fh.core.num_stages * sizeof(FZMStageInfo));
        if (!file) {
            throw std::runtime_error(
                "Failed to read stage data from file: " + filename +
                " (" + std::to_string(fh.core.num_stages) + " stages × " +
                std::to_string(sizeof(FZMStageInfo)) + " bytes each)");
        }
    }

    fh.buffers.resize(fh.core.num_buffers);
    if (fh.core.num_buffers > 0) {
        file.read(reinterpret_cast<char*>(fh.buffers.data()),
                  fh.core.num_buffers * sizeof(FZMBufferEntry));
        if (!file) {
            throw std::runtime_error(
                "Failed to read buffer data from file: " + filename +
                " (" + std::to_string(fh.core.num_buffers) + " buffers × " +
                std::to_string(sizeof(FZMBufferEntry)) + " bytes each)");
        }
    }
    file.close();

    // ── Step 4: verify header checksum (v3.1+ exact-version match only) ───────
    // Skip verification for legacy files and minor-version-mismatched files:
    // the checksum semantics may differ if the reader and writer don't agree
    // on the exact header layout.
    if (!is_legacy && version_exact &&
        (fh.core.flags & FZM_FLAG_HAS_HEADER_CHECKSUM)) {
        uint32_t expected = fh.core.header_checksum;
        uint32_t computed = computeHeaderChecksum(fh);
        if (computed != expected) {
            throw std::runtime_error(
                "FZM header checksum mismatch in '" + filename +
                "': expected 0x" + [](uint32_t v) {
                    char buf[9]; snprintf(buf, sizeof(buf), "%08X", v); return std::string(buf);
                }(expected) + ", computed 0x" + [](uint32_t v) {
                    char buf[9]; snprintf(buf, sizeof(buf), "%08X", v); return std::string(buf);
                }(computed) + " — file may be corrupted"
            );
        }
        FZ_LOG(DEBUG, "FZM header checksum OK (0x%08X)", computed);
    }

    FZ_LOG(INFO, "Read FZM header from %s (v%u.%u, %.2f MB uncompressed, %.2f MB compressed, %u stages, %u buffers, %u source(s))",
           filename.c_str(), file_major, file_minor,
           fh.core.uncompressed_size / (1024.0 * 1024.0),
           fh.core.compressed_size / (1024.0 * 1024.0),
           fh.core.num_stages, fh.core.num_buffers, fh.core.num_sources);

    return fh;
}

void* Pipeline::loadCompressedData(
    const std::string& filename,
    const FZMFileHeader& fh,
    cudaStream_t stream,
    MemoryPool* pool
) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    // Seek past the variable-length header using header_size
    file.seekg(fh.core.header_size, std::ios::beg);
    if (!file) {
        throw std::runtime_error(
            "Failed to seek past header in file: " + filename +
            " (header_size=" + std::to_string(fh.core.header_size) + ")");
    }

    size_t data_size = fh.core.compressed_size;
    void* h_data = malloc(data_size);
    if (!h_data) {
        throw std::runtime_error(
            "Failed to allocate host buffer for compressed data (" +
            std::to_string(data_size) + " bytes) from file: " + filename);
    }

    file.read(reinterpret_cast<char*>(h_data), data_size);
    if (!file) {
        free(h_data);
        throw std::runtime_error(
            "Failed to read compressed data from file: " + filename +
            " (expected " + std::to_string(data_size) + " bytes)");
    }
    file.close();

    // Verify data checksum while the payload is still in h_data (host memory).
    if (fh.core.flags & FZM_FLAG_HAS_DATA_CHECKSUM) {
        uint32_t computed = crc32(h_data, data_size);
        if (computed != fh.core.data_checksum) {
            free(h_data);
            throw std::runtime_error(
                "FZM data checksum mismatch in '" + filename +
                "': expected 0x" + [](uint32_t v) {
                    char buf[9]; snprintf(buf, sizeof(buf), "%08X", v); return std::string(buf);
                }(fh.core.data_checksum) + ", computed 0x" + [](uint32_t v) {
                    char buf[9]; snprintf(buf, sizeof(buf), "%08X", v); return std::string(buf);
                }(computed) + " — compressed data may be corrupted"
            );
        }
        FZ_LOG(DEBUG, "FZM data checksum OK (0x%08X, %.2f MB)",
               computed, data_size / (1024.0 * 1024.0));
    }

    void* d_data = nullptr;
    cudaError_t err;
    if (pool) {
        d_data = pool->allocate(data_size, stream, "compressed_data_load");
        if (!d_data) {
            free(h_data);
            throw std::runtime_error(
                "Pool allocation failed for compressed data load (" +
                std::to_string(data_size) + " bytes); pool may be exhausted — "
                "pass a larger pool_override_bytes to decompressFromFile()");
        }
    } else {
        err = cudaMalloc(&d_data, data_size);
        if (err != cudaSuccess) {
            free(h_data);
            throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
    }

    err = cudaMemcpyAsync(d_data, h_data, data_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        if (pool) pool->free(d_data, stream);
        else cudaFree(d_data);
        free(h_data);
        throw std::runtime_error("cudaMemcpyAsync failed: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        if (pool) pool->free(d_data, stream);
        else cudaFree(d_data);
        free(h_data);
        throw std::runtime_error("cudaStreamSynchronize failed: " + std::string(cudaGetErrorString(err)));
    }

    free(h_data);

    FZ_LOG(DEBUG, "Loaded %.2f MB compressed data to GPU", data_size / (1024.0 * 1024.0));

    return d_data;
}

void Pipeline::decompressFromFile(
    const std::string& filename,
    void** d_output,
    size_t* output_size,
    cudaStream_t stream,
    PipelinePerfResult* perf_out,
    size_t pool_override_bytes
) {
    FZ_LOG(INFO, "Decompressing from file: %s", filename.c_str());

    // 1. Read header
    FZMFileHeader fh = readHeader(filename);
    FZ_LOG(DEBUG, "Header: %u stages, %u buffers, %.2f MB compressed, %.2f MB uncompressed",
           fh.core.num_stages, fh.core.num_buffers,
           fh.core.compressed_size / (1024.0 * 1024.0),
           fh.core.uncompressed_size / (1024.0 * 1024.0));

    // 2. Local memory pool for GPU temporaries.
    //
    // Pool sizing rationale:
    //   At any point during inverse-DAG execution with MINIMAL strategy the
    //   simultaneously live allocations are:
    //     (a) the compressed blob loaded from disk:  C bytes  (freed at very end)
    //     (b) the input buffers of the current level: ≤ max_stage_uncompressed
    //     (c) the output buffers of the current level: ≤ max_stage_uncompressed
    //
    //   Parallel branches at a single level share the same data (they divide it),
    //   so their combined output is bounded by max_stage_uncompressed, not
    //   num_branches × max_stage_uncompressed.
    //
    //   Tight formula:  C + 2 * max_stage_uncompressed
    //   With 25% headroom: C + 2.5 * max_stage_uncompressed
    //
    //   This is significantly smaller than the old (C + U) * 2 = 2C + 2U when
    //   compression is effective (C << U), while remaining safe.
    size_t pool_size_bytes;
    if (pool_override_bytes > 0) {
        pool_size_bytes = pool_override_bytes;
        FZ_LOG(INFO, "decompressFromFile: using caller-supplied pool size of %.2f MB",
               pool_size_bytes / (1024.0 * 1024.0));
    } else {
        // Compute max intermediate expansion across all pipeline-output buffers.
        // fh.buffers[i].uncompressed_size is the decompressed size of that stage's
        // output, i.e. the maximum size any single level boundary can reach.
        size_t max_stage_uncompressed = static_cast<size_t>(fh.core.uncompressed_size);
        for (const auto& buf : fh.buffers) {
            max_stage_uncompressed = std::max(
                max_stage_uncompressed,
                static_cast<size_t>(buf.uncompressed_size));
        }
        const size_t C = static_cast<size_t>(fh.core.compressed_size);
        // 2.5× intermediate headroom + fixed CUDA mempool overhead guard (32 MB)
        pool_size_bytes = C
                        + static_cast<size_t>(2.5 * static_cast<double>(max_stage_uncompressed))
                        + 32ULL * 1024 * 1024;
        FZ_LOG(DEBUG, "decompressFromFile: pool sizing: C=%.2f MB, max_stage_U=%.2f MB "
               "-> pool=%.2f MB (old formula would have been %.2f MB)",
               C / (1024.0 * 1024.0),
               max_stage_uncompressed / (1024.0 * 1024.0),
               pool_size_bytes / (1024.0 * 1024.0),
               (C + fh.core.uncompressed_size) * 2.0 / (1024.0 * 1024.0));
    }
    MemoryPoolConfig local_pool_cfg(pool_size_bytes, 1.0f);
    MemoryPool local_pool(local_pool_cfg);

    // 3. Load compressed blob to GPU
    void* d_compressed = loadCompressedData(filename, fh, stream, &local_pool);

    try {
        // ── Reconstruct stages from file header (forward order: level 0 first) ──
        // Stages are stored in forward execution order by buildHeader().
        std::vector<std::unique_ptr<Stage>> owned_stages;
        owned_stages.reserve(fh.core.num_stages);
        for (uint32_t i = 0; i < fh.core.num_stages; i++) {
            const auto& si = fh.stages[i];
            Stage* stage = createStage(si.stage_type, si.stage_config, si.config_size);
            stage->setInverse(true);
            owned_stages.emplace_back(stage);
            FZ_LOG(DEBUG, "Reconstructed inverse stage %u: %s", i,
                   stageTypeToString(si.stage_type).c_str());
        }

        // ── Build forward-topology description ────────────────────────────────
        std::vector<FwdStageDesc> fwd_topology;
        fwd_topology.reserve(fh.core.num_stages);
        for (uint32_t i = 0; i < fh.core.num_stages; i++) {
            const auto& si = fh.stages[i];
            FwdStageDesc d;
            d.stage = owned_stages[i].get();
            for (int j = 0; j < si.num_outputs && j < static_cast<int>(FZM_MAX_STAGE_OUTPUTS); j++) {
                if (si.output_buffer_ids[j] != 0xFFFF)
                    d.output_buf_ids.push_back(static_cast<int>(si.output_buffer_ids[j]));
            }
            for (int j = 0; j < si.num_inputs && j < static_cast<int>(FZM_MAX_STAGE_INPUTS); j++) {
                if (si.input_buffer_ids[j] != 0xFFFF)
                    d.input_buf_ids.push_back(static_cast<int>(si.input_buffer_ids[j]));
            }
            fwd_topology.push_back(std::move(d));
        }

        // ── Build pipeline-output map: fwd_buf_id → {device ptr, size} ────────
        PipelineOutputMap po_map;
        for (const auto& entry : fh.buffers) {
            void* d_buf = static_cast<uint8_t*>(d_compressed) + entry.byte_offset;
            po_map[static_cast<int>(entry.dag_buffer_id)] =
                {d_buf, static_cast<size_t>(entry.data_size)};
        }

        // ── Build per-source uncompressed sizes from header (v3+) ─────────────
        // Source stages are identified by topology: a stage is a source when
        // none of its input buffer IDs are produced as outputs by another stage.
        // Sizes come from fh.core.source_uncompressed_sizes[] (written by
        // buildHeader in the same discovery order); v2-file fallback uses the
        // total uncompressed_size (correct for single-source pipelines).
        std::unordered_map<Stage*, size_t> source_sizes;
        {
            std::unordered_set<int> all_output_ids;
            for (const auto& d : fwd_topology) {
                for (int bid : d.output_buf_ids) all_output_ids.insert(bid);
            }
            uint16_t source_idx = 0;
            for (const auto& fwd_desc : fwd_topology) {
                bool is_source = true;
                for (int in_bid : fwd_desc.input_buf_ids) {
                    if (all_output_ids.count(in_bid)) { is_source = false; break; }
                }
                if (!is_source) continue;
                size_t sz = (fh.core.num_sources > source_idx &&
                             fh.core.source_uncompressed_sizes[source_idx] > 0)
                            ? static_cast<size_t>(fh.core.source_uncompressed_sizes[source_idx])
                            : static_cast<size_t>(fh.core.uncompressed_size);
                source_sizes[fwd_desc.stage] = sz;
                ++source_idx;
            }
        }
        if (source_sizes.empty()) {
            throw std::runtime_error(
                "decompressFromFile: could not identify any source stage in topology");
        }

        // ── Build and finalize the inverse DAG ───────────────────────────────
        bool do_profile = (perf_out != nullptr);
        auto [inv_dag, inv_result_map] = buildInverseDAG(
            fwd_topology, po_map, &local_pool, MemoryStrategy::MINIMAL,
            source_sizes, do_profile);

        FZ_LOG(DEBUG, "Inverse DAG (file): %zu levels, max_parallelism=%d",
               inv_dag->getLevels().size(), inv_dag->getMaxParallelism());

        // ── Execute ───────────────────────────────────────────────────────────
        auto t_compute_start = std::chrono::steady_clock::now();
        inv_dag->execute(stream);
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        for (auto& stage_ptr : owned_stages) {
            stage_ptr->postStreamSync(stream);
        }

        auto stage_timings = do_profile
            ? inv_dag->collectTimings()
            : std::vector<StageTimingResult>{};

        // ── Extract result ────────────────────────────────────────────────────
        // Find the source stage and its inverse result buffer.
        // For the file path we handle only the primary (first-found) source stage.
        Stage* fwd_source = nullptr;
        for (const auto& fwd_desc : fwd_topology) {
            if (source_sizes.count(fwd_desc.stage)) {
                fwd_source = fwd_desc.stage;
                break;
            }
        }
        if (!fwd_source) {
            throw std::runtime_error(
                "decompressFromFile: no source stage found in reconstructed topology");
        }
        int    inv_result_buf_id = inv_result_map.at(fwd_source);
        void*  d_inv_result      = inv_dag->getBuffer(inv_result_buf_id);
        size_t actual_size       = inv_dag->getBufferSize(inv_result_buf_id);
        // Prefer the stage's post-execution reported size over the estimated DAG size.
        {
            auto post_sizes = fwd_source->getActualOutputSizesByName();
            auto post_names = fwd_source->getOutputNames();
            if (!post_names.empty() && post_sizes.count(post_names[0])) {
                actual_size = post_sizes.at(post_names[0]);
            }
        }

        void* d_final = nullptr;
        cudaError_t err = cudaMalloc(&d_final, actual_size);
        if (err != cudaSuccess) {
            inv_dag->reset(stream);
            local_pool.free(d_inv_result, stream);
            throw std::runtime_error(
                "cudaMalloc for decompressed output failed (" +
                std::to_string(actual_size) + " bytes): " +
                cudaGetErrorString(err));
        }
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_final, d_inv_result, actual_size,
                                      cudaMemcpyDeviceToDevice, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        auto t_compute_end = std::chrono::steady_clock::now();
        float compute_ms = std::chrono::duration<float, std::milli>(
            t_compute_end - t_compute_start).count();

        // ── Cleanup ───────────────────────────────────────────────────────────
        inv_dag->reset(/*stream=*/0);
        local_pool.free(d_inv_result, /*stream=*/0);
        local_pool.free(d_compressed, /*stream=*/0);

        *d_output    = d_final;
        *output_size = actual_size;

        // ── Perf result ───────────────────────────────────────────────────────
        if (perf_out) {
            auto log_levels = buildLevelTimings(stage_timings);
            float dag_ms = 0.0f;
            for (const auto& lv : log_levels) dag_ms += lv.elapsed_ms;
            if (dag_ms <= 0.0f) dag_ms = compute_ms;

            perf_out->is_compress     = false;
            perf_out->host_elapsed_ms = compute_ms;
            perf_out->dag_elapsed_ms  = dag_ms;
            perf_out->input_bytes     = fh.core.compressed_size;
            perf_out->output_bytes    = actual_size;
            perf_out->stages          = std::move(stage_timings);
            perf_out->levels          = std::move(log_levels);
        }

        float dag_tput  = static_cast<float>(actual_size) / (
            (perf_out ? perf_out->dag_elapsed_ms : compute_ms) * 1e-3f) / 1e9f;
        float pipe_tput = static_cast<float>(actual_size) / (compute_ms * 1e-3f) / 1e9f;
        FZ_LOG(INFO,
               "Decompression complete (DAG-native, file): "
               "%.2f MB -> %zu bytes (compute=%.2f ms, DAG=%.2f GB/s, pipeline=%.2f GB/s)",
               fh.core.compressed_size / (1024.0 * 1024.0), actual_size,
               compute_ms, dag_tput, pipe_tput);

    } catch (...) {
        // Ensure all in-flight GPU ops complete before freeing pool memory.
        cudaStreamSynchronize(stream);
        local_pool.free(d_compressed, /*stream=*/0);
        throw;
    }
}

// ── Instance decompressFromFile (respects setPoolManagedDecompOutput) ─────────

void Pipeline::decompressFromFileInstance(
    const std::string&  filename,
    void**              d_output,
    size_t*             output_size,
    cudaStream_t        stream,
    PipelinePerfResult* perf_out
) {
    // Delegate all the heavy lifting to the static overload (always gives us a
    // fresh cudaMalloc'd caller-owned pointer).
    void*  d_tmp  = nullptr;
    size_t tmp_sz = 0;
    Pipeline::decompressFromFile(filename, &d_tmp, &tmp_sz, stream, perf_out);

    if (!pool_managed_decomp_) {
        // Caller-owned: hand the cudaMalloc'd pointer straight through.
        *d_output    = d_tmp;
        *output_size = tmp_sz;
        return;
    }

    // Pool-owned: allocate from our pool and D2D-copy, then free the temp.
    // Free any previous pool-managed decompress output first.
    for (void* p : d_decomp_outputs_) {
        if (p && mem_pool_) mem_pool_->free(p, stream);
    }
    d_decomp_outputs_.clear();

    void* d_pool = mem_pool_->allocate(tmp_sz, stream, "decomp_output_file",
                                        /*persistent=*/true);
    if (!d_pool) {
        cudaFree(d_tmp);
        throw std::runtime_error(
            "decompressFromFile (pool-owned): pool allocation failed (" +
            std::to_string(tmp_sz) + " bytes); pool may be exhausted");
    }

    FZ_CUDA_CHECK(cudaMemcpyAsync(d_pool, d_tmp, tmp_sz,
                                   cudaMemcpyDeviceToDevice, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_tmp);

    d_decomp_outputs_.push_back(d_pool);
    *d_output    = d_pool;
    *output_size = tmp_sz;

    FZ_LOG(INFO, "decompressFromFile (pool-owned): %.2f MB at %p",
           tmp_sz / (1024.0 * 1024.0), d_pool);
}

} // namespace fz
