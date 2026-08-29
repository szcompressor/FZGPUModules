#pragma once

/**
 * @file tee_stage.h
 * @brief TeeStage — duplicate one input into N identical outputs, and back.
 *
 * ## Why a Tee stage exists
 *
 * `Pipeline::compress()` allows exactly one "source" node (a stage with no
 * forward producer, consuming the pipeline's raw input directly — see its
 * `input_nodes_.size() != 1` check). Some pipelines genuinely need the SAME
 * raw data to reach two independent downstream consumers (e.g. a stage that
 * needs both the original data AND a value computed several stages later, to
 * verify or correct a lossy transform's reconstruction against it — see
 * `Cdf97OutlierCorrectStage`). `TeeStage` is the single legal source in such
 * a pipeline, duplicating its one input into N branches.
 *
 * ## The round-trip contract this satisfies
 *
 * Every FZGM stage must satisfy: `inverse input count == forward output
 * count`, and `inverse output count == forward input count`, with inverse
 * output `k` reconstructing forward input `k` (`buildInverseDAG()` wires
 * strictly by this contract — see `MergeStage`'s doc comment for the same
 * point from the N-to-1 side). `TeeStage` is `1 -> N` forward, so its
 * inverse is `N -> 1`: it receives N candidate reconstructions of its
 * original single input (one from each branch's own inverse chain) and must
 * produce exactly ONE output, reconstructing that original input.
 *
 * Which of the N candidates is authoritative is exactly what
 * `setPassthroughIndex()` picks — inverse is `output = inputs[passthrough]`,
 * the other N-1 candidates are computed (their branches still have to run,
 * since intermediate stages in those branches may be needed elsewhere) but
 * discarded here. For `Cdf97OutlierCorrectStage`'s use, that's the branch
 * that applied the correction, not the plain (uncorrected) transform branch.
 *
 * Byte-transparent (`DataType::UNKNOWN` on every port): a Tee never
 * interprets its data, so it must not gate `finalize()`'s type checking
 * either forward or inverse.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

class TeeStage : public Stage {
public:
    TeeStage() = default;
    ~TeeStage() override = default;

    /// Number of forward output branches (>= 2; 1 would be a no-op copy).
    /// Must be called before `connect()` so `getNumOutputs()` reports right.
    void setNumOutputs(int n) {
        if (n < 2) throw std::runtime_error("TeeStage: setNumOutputs needs >= 2");
        if (n > (int)kMaxBranches)
            throw std::runtime_error("TeeStage: too many branches (max " + std::to_string(kMaxBranches) + ")");
        n_ = n;
    }
    int getNumBranches() const { return n_; }

    /// Inverse: which of the N inverse inputs is the authoritative
    /// reconstruction of the original single forward input. Default 0.
    void setPassthroughIndex(int idx) { passthrough_idx_ = idx; }
    int  getPassthroughIndex() const { return passthrough_idx_; }

    // ── Stage control ─────────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    /// Pure stream-ordered D2D memcpy in both directions — no host sync.
    bool isGraphCompatible() const override { return true; }

    // ── Port model ────────────────────────────────────────────────────────────
    // Forward: 1 input -> n_ outputs. Inverse: n_ inputs -> 1 output.
    size_t getNumInputs()  const override { return is_inverse_ ? (size_t)n_ : 1; }
    size_t getNumOutputs() const override { return is_inverse_ ? 1 : (size_t)n_; }
    std::vector<std::string> getOutputNames() const override {
        if (is_inverse_) return {"output"};
        std::vector<std::string> names;
        for (int i = 0; i < n_; ++i) names.push_back("out" + std::to_string(i));
        return names;
    }

    std::string getName() const override { return "Tee"; }
    uint16_t getStageTypeId() const override { return static_cast<uint16_t>(StageType::TEE); }

    // Byte-transparent — opt out of finalize()'s type checking on every port.
    uint8_t getOutputDataType(size_t) const override { return static_cast<uint8_t>(DataType::UNKNOWN); }
    uint8_t getInputDataType(size_t) const override  { return static_cast<uint8_t>(DataType::UNKNOWN); }

    // ── Execution ─────────────────────────────────────────────────────────────
    void execute(cudaStream_t stream, MemoryPool* pool,
                 const std::vector<void*>& inputs,
                 const std::vector<void*>& outputs,
                 const std::vector<size_t>& sizes) override;

    // ── Size estimation ───────────────────────────────────────────────────────
    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes) const override {
        if (is_inverse_) {
            if (passthrough_idx_ < 0 || passthrough_idx_ >= (int)input_sizes.size()) return {0};
            return {input_sizes[passthrough_idx_]};
        }
        std::vector<size_t> out(n_, input_sizes.empty() ? 0 : input_sizes[0]);
        return out;
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        if (is_inverse_) return {{"output", actual_output_size_}};
        std::unordered_map<std::string, size_t> m;
        for (int i = 0; i < n_; ++i) m["out" + std::to_string(i)] = actual_output_size_;
        return m;
    }
    size_t getActualOutputSize(int index) const override {
        if (is_inverse_) return index == 0 ? actual_output_size_ : 0;
        return (index >= 0 && index < n_) ? actual_output_size_ : 0;
    }

    // ── Serialization ─────────────────────────────────────────────────────────
    size_t serializeHeader(size_t, uint8_t* buf, size_t max_size) const override {
        if (max_size < 8) throw std::runtime_error("TeeStage: header buffer too small");
        int32_t n = n_, p = passthrough_idx_;
        std::memcpy(buf, &n, 4); std::memcpy(buf + 4, &p, 4);
        return 8;
    }
    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size < 8) throw std::runtime_error("TeeStage: header too small");
        int32_t n, p;
        std::memcpy(&n, buf, 4); std::memcpy(&p, buf + 4, 4);
        n_ = n; passthrough_idx_ = p;
    }
    size_t getMaxHeaderSize(size_t) const override { return 8; }

    void saveState()    override { saved_n_ = n_; saved_passthrough_ = passthrough_idx_; }
    void restoreState() override { n_ = saved_n_; passthrough_idx_ = saved_passthrough_; }

private:
    static constexpr size_t kMaxBranches = 8;

    bool is_inverse_ = false;
    int  n_ = 2;
    int  passthrough_idx_ = 0;
    size_t actual_output_size_ = 0;

    int saved_n_ = 2, saved_passthrough_ = 0;
};

} // namespace fz
