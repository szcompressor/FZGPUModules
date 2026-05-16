#pragma once

/**
 * @file huffman_stage.h
 * @brief Huffman entropy coding stage (PHF coarse-grained encoding).
 *
 * Forward: `T[]` → variable-length PHF bitstream (inline phf_header prepended).
 * Inverse: PHF bitstream → `T[]`.
 *
 * Not graph-compatible: forward execute contains two host-synchronous operations
 * (histogram D2H and GPU_coarse_encode sync inside phf::high_level::encode).
 *
 * Supported input types: `uint8_t`, `uint16_t`, `uint32_t`.
 *
 * Serialized header layout (11 bytes):
 *   [0]     DataType of T   (1 byte)
 *   [1..2]  bklen_          (uint16_t LE)
 *   [3..10] original_len_   (uint64_t LE, element count)
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "coders/huffman/phf/hf.h"   // phf_header, phf_stream_t

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

// Forward-declare phf::Buf<T> to avoid pulling hf_buf.h (CUDA-only) into this header.
// HuffmanStage<T>::~HuffmanStage() is defined in huffman_stage.cu where the type is complete.
namespace phf { template<typename E> struct Buf; }

namespace fz {

/**
 * Huffman entropy coding stage.
 *
 * Forward: `T[] → uint8_t[]`  PHF-encoded bitstream with embedded phf_header.
 * Inverse: `uint8_t[] → T[]`  Decoded symbol stream.
 *
 * @tparam T  Input element type: `uint8_t`, `uint16_t`, or `uint32_t`.
 */
template <typename T>
class HuffmanStage : public Stage {
    static_assert(
        std::is_same_v<T, uint8_t>  ||
        std::is_same_v<T, uint16_t> ||
        std::is_same_v<T, uint32_t>,
        "HuffmanStage: T must be uint8_t, uint16_t, or uint32_t.");

public:
    HuffmanStage();
    ~HuffmanStage() override;

    // ── Configuration ─────────────────────────────────────────────────────────

    /**
     * Set the Huffman codebook length (number of distinct symbols).
     *
     * Must be ≤ 2^(8*sizeof(T)).  Typical values:
     *   uint8_t  : 256  (covers all possible byte values)
     *   uint16_t : 1024 (covers quantization codes in [-512, 511])
     *   uint32_t : 1024 (must be set explicitly; 2^32 is too large for a codebook)
     *
     * Set before the first compress() call.  Changing bklen after the first
     * execute() forces Buf reallocation on the next call (old buffers returned to
     * pool, new ones allocated).  Buf is also reallocated when inlen grows past
     * the previously allocated capacity; shrinking inlen reuses the existing
     * allocation.
     * Default: 256 for uint8_t, 1024 for uint16_t/uint32_t.
     */
    void     setBklen(uint32_t bklen) { bklen_ = bklen; }
    uint32_t getBklen() const         { return bklen_; }

    // ── Stage control ─────────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    // Two host-synchronous ops in forward execute make this graph-incompatible.
    bool isGraphCompatible() const override { return false; }

    // ── Pool lifecycle ────────────────────────────────────────────────────────

    /**
     * Called by Pipeline::finalize() after buffer-size propagation.
     *
     * Pre-allocates phf::Buf<T> from the pool using the estimated input size so
     * PREALLOCATE mode commits all memory at finalize time.  If estimated_inlen
     * is 0 (no size hint available), allocation is deferred to the first execute()
     * call.
     */
    void onFinalize(size_t estimated_inlen, MemoryPool* pool) override;

    size_t estimateDeviceFootprintBytes(size_t inlen) const override;
    size_t estimatePinnedFootprintBytes(size_t inlen) const override;

    // ── Execution ─────────────────────────────────────────────────────────────
    void execute(
        cudaStream_t stream,
        MemoryPool*  pool,
        const std::vector<void*>&   inputs,
        const std::vector<void*>&   outputs,
        const std::vector<size_t>&  sizes
    ) override;

    // ── Metadata ──────────────────────────────────────────────────────────────
    std::string getName()      const override { return "Huffman"; }
    size_t      getNumInputs() const override { return 1; }
    size_t      getNumOutputs()const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (input_sizes.empty()) return {0};
        if (!is_inverse_) {
            // Upper bound: generous 2× input for worst-case bitstream + header overhead.
            return {input_sizes[0] * 2 + 4096};
        }
        // Inverse: exact decoded size restored from the serialized header.
        return {original_len_ * sizeof(T)};
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }

    size_t getActualOutputSize(int index) const override {
        return (index == 0) ? actual_output_size_ : 0;
    }

    // ── Type system ───────────────────────────────────────────────────────────
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::HUFFMAN);
    }

    // Byte-transparent output: opt out of pipeline type-compatibility checking.
    uint8_t getOutputDataType(size_t /*output_index*/) const override {
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }
    uint8_t getInputDataType(size_t /*input_index*/) const override {
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }

    // ── Serialization ─────────────────────────────────────────────────────────
    size_t serializeHeader(
        size_t /*output_index*/, uint8_t* buf, size_t max_size
    ) const override {
        if (max_size < 11) return 0;
        buf[0] = static_cast<uint8_t>(dataTypeOf<T>());
        uint16_t bk = static_cast<uint16_t>(bklen_);
        std::memcpy(buf + 1, &bk,           sizeof(uint16_t));
        std::memcpy(buf + 3, &original_len_, sizeof(uint64_t));
        return 11;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size >= 3) {
            uint16_t bk;
            std::memcpy(&bk, buf + 1, sizeof(uint16_t));
            bklen_ = bk;
        }
        if (size >= 11)
            std::memcpy(&original_len_, buf + 3, sizeof(uint64_t));
    }

    size_t getMaxHeaderSize(size_t /*output_index*/) const override { return 11; }

    void saveState() override {
        saved_bklen_        = bklen_;
        saved_original_len_ = original_len_;
        saved_output_size_  = actual_output_size_;
    }

    void restoreState() override {
        bklen_              = saved_bklen_;
        original_len_       = saved_original_len_;
        actual_output_size_ = saved_output_size_;
    }

private:
    bool     is_inverse_         = false;
    uint32_t bklen_              = defaultBklen();
    uint64_t original_len_       = 0;   // element count set by forward execute
    size_t   actual_output_size_ = 0;
    size_t   cap_inlen_          = 0;   // allocated capacity (elements); grow-only
    uint32_t last_bklen_         = 0;   // bklen_ when buf_ was last allocated

    // Histogram launch params — computed once in initBuf(), reused every execute()
    int hist_grid_dim_    = 0;
    int hist_block_dim_   = 0;
    int hist_shmem_use_   = 0;
    int hist_r_per_block_ = 0;

    // PHF working buffers — allocated from pool_ on first execute() or in onFinalize()
    std::unique_ptr<phf::Buf<T>> buf_;
    phf_header header_ {};

    // Pool used for buf_ allocations.  Set by onFinalize() or captured from the
    // pool parameter on the first execute() call.  Raw non-owning pointer; the
    // pool outlives the stage when used inside a Pipeline.
    MemoryPool* pool_ = nullptr;

    // saveState / restoreState snapshots
    uint32_t saved_bklen_        = defaultBklen();
    uint64_t saved_original_len_ = 0;
    size_t   saved_output_size_  = 0;

    static constexpr uint32_t defaultBklen() {
        if constexpr (std::is_same_v<T, uint8_t>) return 256;
        return 1024;
    }

    template<typename U>
    static constexpr DataType dataTypeOf() {
        if constexpr (std::is_same_v<U,  uint8_t>)  return DataType::UINT8;
        if constexpr (std::is_same_v<U, uint16_t>)  return DataType::UINT16;
        return DataType::UINT32;
    }

    // Allocates buf_ from pool and computes histogram launch params for the given
    // element count.  Must be in huffman_stage.cu: calls cudaFuncSetAttribute with
    // a __global__ pointer.  If buf_ already exists, destroys it first (returning
    // its allocations to the pool) before creating the new one.
    void initBuf(size_t inlen, MemoryPool* pool);
};

extern template class HuffmanStage<uint8_t>;
extern template class HuffmanStage<uint16_t>;
extern template class HuffmanStage<uint32_t>;

} // namespace fz
