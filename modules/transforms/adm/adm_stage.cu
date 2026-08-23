// Adapted from the MANS project (Huang et al.), BSD-3-Clause License — see THIRD_PARTY.md.
// Original: nv/adm/ in https://github.com/hpdps-group/MANS
//
// This file implements the ADMStage pool-management and execute() wrapper.
// The actual encode/decode kernels are in mapping_uint16.cu / mapping_uint32.cu.

#include "transforms/adm/adm_stage.h"
#include "stage/stage_registry.h"
#include "transforms/adm/adm_kernels.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "backend/api.h"
#include <stdexcept>
#include <string>

namespace fz {

// ── Helpers ───────────────────────────────────────────────────────────────────

int ADMStage::maxSignalBytes() const {
    return dtype_ == ADMDtype::U16 ? adm::kMaxSignalBytesU16 : adm::kMaxSignalBytesU32;
}

size_t ADMStage::centerElemBytes() const {
    return dtype_ == ADMDtype::U16 ? sizeof(uint16_t) : sizeof(uint32_t);
}

// ── initScratch ───────────────────────────────────────────────────────────────

void ADMStage::initScratch(size_t num_elements, MemoryPool* pool)
{
    const size_t gsize       = adm::adm_gsize(num_elements);
    const size_t flags_words = adm::adm_flags_words(gsize);
    const int    sig_bytes   = maxSignalBytes();
    const size_t ctr_bytes   = centerElemBytes();

    // The decoupled-look-back kernel now groups kWarpsPerBlock warps into one
    // CUDA thread block (see adm_kernels.h). When gsize isn't a multiple of
    // kWarpsPerBlock, the last CUDA block launches a few "phantom" warps whose
    // global warp index is >= gsize; their data-dependent loops are already
    // bounds-checked against data_size and naturally produce max_bits=0, but
    // they still perform unconditional vectorised writes (signal_length[warp],
    // centers[warp], cmpOffset[warp], and the int4 store into d_codes_) at
    // indices up to num_blocks*kWarpsPerBlock-1. gsize_padded rounds every
    // per-warp buffer up to that bound so those writes land in harmless
    // padding instead of out-of-bounds memory; host-side readbacks still only
    // copy the first (real) gsize entries, so this is transparent downstream.
    const size_t num_blocks    = adm::adm_num_blocks(gsize);
    const size_t gsize_padded  = num_blocks * static_cast<size_t>(adm::kWarpsPerBlock);

    // The ADM map kernels each write a full `kBlockElems` slice per grid
    // block (32 lanes × 16 bytes via vectorised int4 stores into d_codes_;
    // 32 lanes × 16 × sig_bytes via int2 stores into the signal buffers),
    // regardless of how many of those elements actually exist in the input.
    // d_codes_, d_concat_signals_, and d_bit_signals_ therefore need to be
    // padded up to a full multiple of kBlockElems (now kBlockElems ×
    // kWarpsPerBlock, per gsize_padded above) or the trailing lanes write
    // past the allocation. Only the first num_elements bytes (resp.
    // num_elements × sig_bytes) are copied to the user output later.
    const size_t padded_n = gsize_padded * static_cast<size_t>(adm::kBlockElems);

    // Release existing allocations before resizing.
    if (d_signal_length_)  pool->freePersistentDevice(d_signal_length_);
    if (d_output_lengths_) pool->freePersistentDevice(d_output_lengths_);
    if (d_centers_)        pool->freePersistentDevice(d_centers_);
    if (d_block_flags_)    pool->freePersistentDevice(d_block_flags_);
    if (d_codes_)          pool->freePersistentDevice(d_codes_);
    if (d_concat_signals_) pool->freePersistentDevice(d_concat_signals_);
    if (d_bit_signals_)    pool->freePersistentDevice(d_bit_signals_);
    if (d_loc_offset_)     pool->freePersistentDevice(d_loc_offset_);
    if (d_prefix_state_)   pool->freePersistentDevice(d_prefix_state_);
    if (d_block_resolved_) pool->freePersistentDevice(d_block_resolved_);
    if (d_overflow_flag_)  pool->freePersistentDevice(d_overflow_flag_);

    d_signal_length_  = static_cast<int*>(pool->allocatePersistentDevice(
        gsize_padded * sizeof(int), "adm.signal_length"));
    d_output_lengths_ = static_cast<int*>(pool->allocatePersistentDevice(
        (gsize_padded + 1) * sizeof(int), "adm.output_lengths"));
    d_centers_        = pool->allocatePersistentDevice(
        gsize_padded * ctr_bytes, "adm.centers");
    d_block_flags_    = static_cast<uint32_t*>(pool->allocatePersistentDevice(
        flags_words * sizeof(uint32_t), "adm.block_flags"));
    d_codes_          = static_cast<uint8_t*>(pool->allocatePersistentDevice(
        padded_n, "adm.codes"));
    d_concat_signals_ = static_cast<uint8_t*>(pool->allocatePersistentDevice(
        padded_n * sig_bytes, "adm.concat_signals"));
    d_bit_signals_    = static_cast<uint8_t*>(pool->allocatePersistentDevice(
        padded_n * sig_bytes, "adm.bit_signals"));
    d_loc_offset_     = static_cast<int*>(pool->allocatePersistentDevice(
        (gsize + 1) * sizeof(int), "adm.loc_offset"));
    d_prefix_state_   = static_cast<int*>(pool->allocatePersistentDevice(
        (gsize + 1) * sizeof(int), "adm.prefix_state"));
    d_block_resolved_ = static_cast<int*>(pool->allocatePersistentDevice(
        (num_blocks + 1) * sizeof(int), "adm.block_resolved"));
    d_overflow_flag_  = static_cast<unsigned int*>(pool->allocatePersistentDevice(
        sizeof(unsigned int), "adm.overflow_flag"));

    cap_elements_ = num_elements;
}

// ── onFinalize ────────────────────────────────────────────────────────────────

void ADMStage::onFinalize(size_t estimated_inlen, MemoryPool* pool)
{
    if (estimated_inlen == 0) return;
    const size_t elem_bytes = (dtype_ == ADMDtype::U16) ? sizeof(uint16_t) : sizeof(uint32_t);
    const size_t n = estimated_inlen / elem_bytes;
    if (n > 0) initScratch(n, pool);
}

// ── estimateDeviceFootprintBytes ──────────────────────────────────────────────

size_t ADMStage::estimateDeviceFootprintBytes(size_t inlen) const
{
    if (inlen == 0) return 0;
    const size_t elem_bytes = (dtype_ == ADMDtype::U16) ? sizeof(uint16_t) : sizeof(uint32_t);
    const size_t n          = inlen / elem_bytes;
    const size_t g          = adm::adm_gsize(n);
    const size_t flags_w    = adm::adm_flags_words(g);
    const size_t sig_bytes  = static_cast<size_t>(maxSignalBytes());
    const size_t ctr_bytes  = centerElemBytes();
    const size_t num_blocks   = adm::adm_num_blocks(g);
    const size_t g_padded     = num_blocks * static_cast<size_t>(adm::kWarpsPerBlock);
    const size_t padded_n     = g_padded * static_cast<size_t>(adm::kBlockElems);

    return g_padded * sizeof(int)        // d_signal_length_
         + (g_padded + 1) * sizeof(int)  // d_output_lengths_
         + g_padded * ctr_bytes          // d_centers_
         + flags_w * sizeof(uint32_t)    // d_block_flags_
         + padded_n                      // d_codes_
         + padded_n * sig_bytes          // d_concat_signals_
         + padded_n * sig_bytes          // d_bit_signals_
         + (g + 1) * sizeof(int)         // d_loc_offset_
         + (g + 1) * sizeof(int)         // d_prefix_state_
         + (num_blocks + 1) * sizeof(int) // d_block_resolved_
         + sizeof(unsigned int);         // d_overflow_flag_
}

size_t ADMStage::estimateScratchBytes(const std::vector<size_t>& input_sizes) const
{
    if (input_sizes.empty()) return 0;
    return estimateDeviceFootprintBytes(input_sizes[0]);
}

// ── estimateOutputSizes ───────────────────────────────────────────────────────

std::vector<size_t> ADMStage::estimateOutputSizes(
    const std::vector<size_t>& input_sizes) const
{
    if (input_sizes.empty()) return {0};
    const size_t inlen = input_sizes[0];
    if (inlen == 0) return {0};

    if (!is_inverse_) {
        const size_t elem_bytes = (dtype_ == ADMDtype::U16) ? sizeof(uint16_t) : sizeof(uint32_t);
        const size_t n = inlen / elem_bytes;
        return { dtype_ == ADMDtype::U16
                 ? adm::get_max_u16_payload_bytes(n)
                 : adm::get_max_u32_payload_bytes(n) };
    }
    // Inverse: output is num_elements_ × sizeof(dtype), set from deserializeHeader().
    if (num_elements_ > 0) {
        const size_t elem_bytes = (dtype_ == ADMDtype::U16) ? sizeof(uint16_t) : sizeof(uint32_t);
        return {num_elements_ * elem_bytes};
    }
    return {inlen};  // conservative fallback if header not yet read
}

// ── execute ───────────────────────────────────────────────────────────────────

void ADMStage::execute(
    cudaStream_t stream,
    MemoryPool*  pool,
    const std::vector<void*>&  inputs,
    const std::vector<void*>&  outputs,
    const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("ADMStage: inputs, outputs, and sizes must be non-empty");

    const size_t byte_size = sizes[0];
    if (byte_size == 0) {
        actual_output_size_ = 0;
        return;
    }

    const size_t elem_bytes = (dtype_ == ADMDtype::U16) ? sizeof(uint16_t) : sizeof(uint32_t);

    if (!is_inverse_) {
        // ── Forward (compress) ────────────────────────────────────────────────
        const size_t num_elements = byte_size / elem_bytes;
        if (num_elements == 0) { actual_output_size_ = 0; return; }

        if (num_elements > cap_elements_)
            initScratch(num_elements, pool);

        adm::AdmScratch s;
        s.d_signal_length  = d_signal_length_;
        s.d_output_lengths = d_output_lengths_;
        s.d_centers        = d_centers_;
        s.d_block_flags    = d_block_flags_;
        s.d_codes          = d_codes_;
        s.d_concat_signals = d_concat_signals_;
        s.d_bit_signals    = d_bit_signals_;
        s.d_loc_offset     = d_loc_offset_;
        s.d_prefix_state   = d_prefix_state_;
        s.d_block_resolved = d_block_resolved_;
        s.d_overflow_flag  = d_overflow_flag_;

        size_t out_size = 0;
        if (dtype_ == ADMDtype::U16) {
            adm::compress_u16(
                static_cast<const uint16_t*>(inputs[0]), num_elements,
                static_cast<uint8_t*>(outputs[0]), out_size, s, stream);
        } else {
            adm::compress_u32(
                static_cast<const uint32_t*>(inputs[0]), num_elements,
                static_cast<uint8_t*>(outputs[0]), out_size, s, stream);
        }
        actual_output_size_ = out_size;
        num_elements_       = num_elements;

    } else {
        // ── Inverse (decompress) ──────────────────────────────────────────────
        if (num_elements_ == 0)
            throw std::runtime_error(
                "ADMStage: num_elements_ is 0 before decompress — "
                "deserializeHeader() must be called first");

        if (num_elements_ > cap_elements_)
            initScratch(num_elements_, pool);

        adm::AdmScratch s;
        s.d_signal_length  = d_signal_length_;
        s.d_output_lengths = d_output_lengths_;
        s.d_centers        = d_centers_;
        s.d_block_flags    = d_block_flags_;
        s.d_codes          = d_codes_;
        s.d_concat_signals = d_concat_signals_;
        s.d_bit_signals    = d_bit_signals_;
        s.d_loc_offset     = d_loc_offset_;
        s.d_prefix_state   = d_prefix_state_;
        s.d_block_resolved = d_block_resolved_;
        s.d_overflow_flag  = d_overflow_flag_;

        if (dtype_ == ADMDtype::U16) {
            adm::decompress_u16(
                static_cast<const uint8_t*>(inputs[0]), byte_size,
                static_cast<uint16_t*>(outputs[0]), num_elements_, s, stream);
        } else {
            adm::decompress_u32(
                static_cast<const uint8_t*>(inputs[0]), byte_size,
                static_cast<uint32_t*>(outputs[0]), num_elements_, s, stream);
        }
        actual_output_size_ = num_elements_ * elem_bytes;
    }
}

} // namespace fz

// Self-registration for FZM-header reconstruction (see stage_registry.h).
FZ_REGISTER_SIMPLE_STAGE(fz::StageType::ADM, fz::ADMStage);
