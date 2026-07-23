#pragma once

/**
 * @file merge_stage.h
 * @brief MergeStage — concatenate N input buffers into one, split back to N.
 *
 * ## Why a Merge stage exists
 *
 * FZGM's DAG runs each stage on its own buffer, and the final `.fzm` archive is
 * assembled from the pipeline's *leaf* outputs **after** all stages have run.
 * That is the right model when each port is compressed independently. But some
 * codecs are defined to run a **single lossless pass over the concatenation of
 * several producer outputs** — most notably cuSZ-Hi's LC back-end, which (in its
 * compression-ratio mode) Huffman-encodes the quant codes and then runs one LC
 * chain over the *contiguous blob* `[Huffman | anchor | outliers]`, and (in its
 * throughput mode) runs one LC chain over `[anchor | outliers]`.
 *
 * There is no way to express "compress the concatenation of these ports" by
 * wiring ports straight into a coder: a coder takes one input buffer, and the
 * archive-level concatenation happens too late (post-compression) to feed a
 * downstream chain. MergeStage fills exactly this gap — it materialises the
 * concatenation as a real buffer *during* the pipeline so a downstream stage can
 * compress it as one stream, and on decompress it splits the reconstructed blob
 * back into the original N segments to feed the inverse producers.
 *
 * It is the structural mirror of the predictors' port asymmetry (e.g.
 * LorenzoQuant is `1 → 3` outputs forward / `3 → 1` inputs inverse); MergeStage
 * is `N → 1` forward / `1 → N` inverse. The existing inverse-DAG builder wires
 * both shapes by input-position index, so MergeStage needs no new DAG support.
 *
 * ## Stream format
 *
 * The forward output is a **pure concatenation** of the inputs in connection
 * order — no in-stream header (matching cuSZ-Hi's byte layout). The per-segment
 * sizes are data-dependent (outlier counts vary), so they are captured at
 * compress time and carried in the FZM stage config header (like RZEStage's
 * `cached_orig_bytes`), which the inverse path uses to split.
 *
 * ## Serialized config header
 *   `[0]`        uint8   num_segments (N)
 *   `[1 .. 4N]`  uint32  segment_size × N  (LE, in connection order)
 *   `[4N+1 ..]`  name table: per segment `[uint8 len][len bytes]`
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

/**
 * Structural N→1 / 1→N concatenation stage.
 *
 * `setSegmentNames({...})` — defines N (= number of inputs to merge / outputs to
 * split into) and the inverse output-port names, in concatenation order. Must be
 * called before `connect()` so `getNumInputs()` reports the right count.
 *
 * Byte-transparent: opts out of type checking (segments are opaque bytes once
 * merged). Pure stream-ordered D2D copies → graph-capturable in both directions.
 */
class MergeStage : public Stage {
public:
    MergeStage() = default;
    ~MergeStage() override = default;

    // ── Configuration ───────────────────────────────────────────────────────
    /// Define the N segments (concatenation order = inverse output order).
    void setSegmentNames(const std::vector<std::string>& names) {
        if (names.empty())
            throw std::runtime_error("MergeStage: at least one segment name required");
        if (names.size() > kMaxSegments)
            throw std::runtime_error("MergeStage: too many segments (max "
                                     + std::to_string(kMaxSegments) + ")");
        segment_names_ = names;
        segment_sizes_.assign(names.size(), 0);
    }
    const std::vector<std::string>& getSegmentNames() const { return segment_names_; }
    size_t getNumSegments() const { return segment_names_.size(); }

    // ── Stage control ───────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    /// Pure stream-ordered D2D memcpy in both directions — no host sync.
    bool isGraphCompatible() const override { return true; }

    // ── Port model ──────────────────────────────────────────────────────────
    // Forward: N inputs → 1 output ("output").
    // Inverse: 1 input  → N outputs (segment_names_).
    size_t getNumInputs()  const override { return is_inverse_ ? 1 : segment_names_.size(); }
    size_t getNumOutputs() const override { return is_inverse_ ? segment_names_.size() : 1; }
    std::vector<std::string> getOutputNames() const override {
        return is_inverse_ ? segment_names_ : std::vector<std::string>{"output"};
    }

    std::string getName() const override { return "Merge"; }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::MERGE);
    }

    // Byte-transparent — opt out of finalize() type checking on every port.
    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }
    uint8_t getInputDataType(size_t) const override {
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }

    // ── Execution ───────────────────────────────────────────────────────────
    void execute(
        fz::stream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    // ── Size estimation ───────────────────────────────────────────────────
    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (is_inverse_) {
            // 1 input (merged blob) → N segment outputs, sizes from the config
            // header (restored via deserializeHeader) or cached from forward.
            return segment_sizes_;
        }
        // N inputs → 1 output = pure concatenation (no in-stream header).
        size_t total = 0;
        for (size_t s : input_sizes) total += s;
        return {total};
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        if (is_inverse_) {
            std::unordered_map<std::string, size_t> m;
            for (size_t i = 0; i < segment_names_.size(); i++)
                m[segment_names_[i]] = (i < segment_sizes_.size()) ? segment_sizes_[i] : 0;
            return m;
        }
        return {{"output", merged_total_}};
    }

    size_t getActualOutputSize(int index) const override {
        if (is_inverse_) {
            if (index < 0 || index >= (int)segment_sizes_.size()) return 0;
            return segment_sizes_[index];
        }
        return (index == 0) ? merged_total_ : 0;
    }

    // ── Serialization ─────────────────────────────────────────────────────
    size_t serializeHeader(size_t, uint8_t* buf, size_t max_size) const override {
        const size_t N = segment_names_.size();
        size_t need = 1 + 4 * N;
        for (const auto& nm : segment_names_) need += 1 + nm.size();
        if (need > max_size) return 0;
        size_t off = 0;
        buf[off++] = static_cast<uint8_t>(N);
        for (size_t i = 0; i < N; i++) {
            uint32_t sz = (i < segment_sizes_.size()) ? static_cast<uint32_t>(segment_sizes_[i]) : 0u;
            std::memcpy(buf + off, &sz, sizeof(uint32_t));
            off += sizeof(uint32_t);
        }
        for (const auto& nm : segment_names_) {
            buf[off++] = static_cast<uint8_t>(nm.size());
            std::memcpy(buf + off, nm.data(), nm.size());
            off += nm.size();
        }
        return off;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size < 1) return;
        size_t off = 0;
        const size_t N = buf[off++];
        segment_sizes_.assign(N, 0);
        for (size_t i = 0; i < N && off + 4 <= size; i++) {
            uint32_t sz = 0;
            std::memcpy(&sz, buf + off, sizeof(uint32_t));
            off += sizeof(uint32_t);
            segment_sizes_[i] = sz;
        }
        segment_names_.assign(N, "");
        for (size_t i = 0; i < N && off < size; i++) {
            const size_t len = buf[off++];
            if (off + len > size) break;
            segment_names_[i].assign(reinterpret_cast<const char*>(buf + off), len);
            off += len;
        }
    }

    size_t getMaxHeaderSize(size_t) const override { return FZM_STAGE_CONFIG_SIZE; }

    void saveState() override {
        saved_names_ = segment_names_;
        saved_sizes_ = segment_sizes_;
    }
    void restoreState() override {
        segment_names_ = saved_names_;
        segment_sizes_ = saved_sizes_;
    }

private:
    static constexpr size_t kMaxSegments = 16;

    bool is_inverse_ = false;
    std::vector<std::string> segment_names_;
    std::vector<size_t>      segment_sizes_;  // per-segment byte sizes (concat order)
    size_t                   merged_total_ = 0;

    std::vector<std::string> saved_names_;
    std::vector<size_t>      saved_sizes_;
};

} // namespace fz
