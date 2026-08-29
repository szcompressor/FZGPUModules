/**
 * @file speck2d_stage.cu
 * @brief Speck2DStage::execute() -- drives the GPU-parallel SPECK-like coder.
 *
 * Per-shape (nx,ny) state (quadtree geometry + its device-resident arrays) is
 * cached across calls in `Impl` (mirrors RLEStage's persistent forward-path
 * scratch) since the host tree build is expensive relative to the device
 * compute (measured ~350-550ms at 3600x1800 -- see memory/speck_gpu_design.md
 * P3.5) and is purely a function of (nx,ny), not the data.
 */

#include "coders/speck2d/speck2d_stage.h"
#include "coders/speck2d/speck2d_kernels.cuh"
#include "cuda_check.h"
#include "stage/stage_registry.h"
#include <cub/cub.cuh>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace fz {

using namespace speck2d;

struct Speck2DStage::Impl {
    // ── per-shape geometry (rebuilt only when (nx,ny) changes) ────────────────
    int nx = 0, ny = 0, nn = 0, nl = 0, maxL = 0, nFused = 0; size_t n = 0;
    int *d_par = nullptr, *d_lf = nullptr, *d_px = nullptr;
    int *d_c0 = nullptr, *d_c1 = nullptr, *d_c2 = nullptr, *d_c3 = nullptr;
    int *d_levelnodes = nullptr, *d_lvl_starts = nullptr, *d_lvl_counts = nullptr, *d_leaves = nullptr;
    std::vector<int> level_start, level_count;

    // ── encode scratch ─────────────────────────────────────────────────────────
    uint32_t* d_mag = nullptr; uint8_t* d_sgn = nullptr; int* d_msb = nullptr;
    int* d_on = nullptr; uint8_t* d_vis = nullptr;
    int *d_bitsA = nullptr, *d_offA = nullptr, *d_bitsB = nullptr, *d_offB = nullptr;
    void* d_tmpA = nullptr; size_t tmpbA = 0;
    void* d_tmpB = nullptr; size_t tmpbB = 0;
    uint32_t* d_out = nullptr; size_t out_words_cap = 0;
    int pend_B = 0, pend_offA = 0, pend_bitsA = 0, pend_offB = 0, pend_bitsB = 0;

    // ── decode scratch ──────────────────────────────────────────────────────────
    uint8_t* d_present = nullptr;
    int *d_flag = nullptr, *d_rank = nullptr, *d_gaps = nullptr, *d_len = nullptr, *d_off = nullptr;
    int *d_bflag = nullptr, *d_brank = nullptr;
    uint64_t* d_ones = nullptr;
    void* d_tmpLevel = nullptr; size_t tmpbLevel = 0;
    void* d_tmpBit = nullptr; size_t tmpbBit = 0;
    uint64_t cap_nbitsA = 0;
    int* d_cursor = nullptr;
    uint8_t* d_stream = nullptr; size_t cap_stream = 0;
    uint32_t* d_decoeff = nullptr; uint8_t* d_desgn = nullptr;

    ~Impl() { freeAll(); }

    // NOTE: must NOT reset via `*this = Impl{}` -- that would construct a
    // temporary, assign it, then destroy the temporary, which (having a
    // user-declared destructor that itself calls freeAll()) recurses without
    // ever terminating. Reset every field explicitly instead.
    void freeAll() {
        void** ptrs[] = {
            (void**)&d_par, (void**)&d_lf, (void**)&d_px, (void**)&d_c0, (void**)&d_c1, (void**)&d_c2, (void**)&d_c3,
            (void**)&d_levelnodes, (void**)&d_lvl_starts, (void**)&d_lvl_counts, (void**)&d_leaves,
            (void**)&d_mag, (void**)&d_sgn, (void**)&d_msb, (void**)&d_on, (void**)&d_vis,
            (void**)&d_bitsA, (void**)&d_offA, (void**)&d_bitsB, (void**)&d_offB, (void**)&d_tmpA, (void**)&d_tmpB, (void**)&d_out,
            (void**)&d_present, (void**)&d_flag, (void**)&d_rank, (void**)&d_gaps, (void**)&d_len, (void**)&d_off,
            (void**)&d_bflag, (void**)&d_brank, (void**)&d_ones, (void**)&d_tmpLevel, (void**)&d_tmpBit,
            (void**)&d_cursor, (void**)&d_stream, (void**)&d_decoeff, (void**)&d_desgn };
        for (void** p : ptrs) { if (*p) FZ_CUDA_CHECK_WARN(cudaFree(*p)); *p = nullptr; }
        nx = ny = nn = nl = maxL = nFused = 0; n = 0;
        level_start.clear(); level_count.clear();
        tmpbA = tmpbB = 0; out_words_cap = 0;
        pend_B = pend_offA = pend_bitsA = pend_offB = pend_bitsB = 0;
        tmpbLevel = tmpbBit = 0; cap_nbitsA = 0; cap_stream = 0;
    }

    bool sameShape(int nx_, int ny_) const { return nx == nx_ && ny == ny_ && n > 0; }

    void ensureShape(int nx_, int ny_) {
        if (sameShape(nx_, ny_)) return;
        freeAll();
        nx = nx_; ny = ny_; n = (size_t)nx * ny;

        Tree t = buildTree(nx, ny);
        nn = t.nnodes(); maxL = t.max_level;
        auto up = [&](int** dp, const std::vector<int>& h) {
            FZ_CUDA_CHECK(cudaMalloc(dp, nn * 4));
            FZ_CUDA_CHECK(cudaMemcpy(*dp, h.data(), nn * 4, cudaMemcpyHostToDevice));
        };
        up(&d_par, t.parent); up(&d_lf, t.is_leaf); up(&d_px, t.pixel);
        up(&d_c0, t.child[0]); up(&d_c1, t.child[1]); up(&d_c2, t.child[2]); up(&d_c3, t.child[3]);

        std::vector<std::vector<int>> bl(maxL + 1);
        for (int i = 0; i < nn; ++i) bl[t.level[i]].push_back(i);
        std::vector<int> flatL;
        level_start.assign(maxL + 1, 0); level_count.assign(maxL + 1, 0);
        for (int L = 0; L <= maxL; ++L) {
            level_start[L] = (int)flatL.size(); level_count[L] = (int)bl[L].size();
            for (int idn : bl[L]) flatL.push_back(idn);
        }
        FZ_CUDA_CHECK(cudaMalloc(&d_levelnodes, flatL.size() * 4));
        FZ_CUDA_CHECK(cudaMemcpy(d_levelnodes, flatL.data(), flatL.size() * 4, cudaMemcpyHostToDevice));
        FZ_CUDA_CHECK(cudaMalloc(&d_lvl_starts, (maxL + 1) * 4));
        FZ_CUDA_CHECK(cudaMalloc(&d_lvl_counts, (maxL + 1) * 4));
        FZ_CUDA_CHECK(cudaMemcpy(d_lvl_starts, level_start.data(), (maxL + 1) * 4, cudaMemcpyHostToDevice));
        FZ_CUDA_CHECK(cudaMemcpy(d_lvl_counts, level_count.data(), (maxL + 1) * 4, cudaMemcpyHostToDevice));
        nFused = chooseShallowLevels(level_count);

        std::vector<int> leaves; for (int i = 0; i < nn; ++i) if (t.is_leaf[i]) leaves.push_back(i);
        nl = (int)leaves.size();
        FZ_CUDA_CHECK(cudaMalloc(&d_leaves, nl * 4));
        FZ_CUDA_CHECK(cudaMemcpy(d_leaves, leaves.data(), nl * 4, cudaMemcpyHostToDevice));

        // encode scratch
        FZ_CUDA_CHECK(cudaMalloc(&d_mag, n * 4)); FZ_CUDA_CHECK(cudaMalloc(&d_sgn, n));
        FZ_CUDA_CHECK(cudaMalloc(&d_msb, n * 4));
        FZ_CUDA_CHECK(cudaMalloc(&d_on, nn * 4)); FZ_CUDA_CHECK(cudaMalloc(&d_vis, nn));
        FZ_CUDA_CHECK(cudaMalloc(&d_bitsA, nn * 4)); FZ_CUDA_CHECK(cudaMalloc(&d_offA, nn * 4));
        FZ_CUDA_CHECK(cudaMalloc(&d_bitsB, nl * 4)); FZ_CUDA_CHECK(cudaMalloc(&d_offB, nl * 4));
        cub::DeviceScan::ExclusiveSum(d_tmpA, tmpbA, d_bitsA, d_offA, nn); FZ_CUDA_CHECK(cudaMalloc(&d_tmpA, tmpbA));
        cub::DeviceScan::ExclusiveSum(d_tmpB, tmpbB, d_bitsB, d_offB, nl); FZ_CUDA_CHECK(cudaMalloc(&d_tmpB, tmpbB));

        // decode scratch (present/onset reuse d_vis/d_on's SIZE but decode needs
        // its own onset array distinct from encode's -- share d_on since encode
        // and decode never run concurrently for one Impl/shape).
        FZ_CUDA_CHECK(cudaMalloc(&d_present, nn));
        FZ_CUDA_CHECK(cudaMalloc(&d_flag, nn * 4)); FZ_CUDA_CHECK(cudaMalloc(&d_rank, nn * 4));
        FZ_CUDA_CHECK(cudaMalloc(&d_len, nl * 4)); FZ_CUDA_CHECK(cudaMalloc(&d_off, nl * 4));
        FZ_CUDA_CHECK(cudaMalloc(&d_decoeff, n * 4)); FZ_CUDA_CHECK(cudaMalloc(&d_desgn, n));
        cub::DeviceScan::ExclusiveSum(d_tmpLevel, tmpbLevel, d_flag, d_rank, nn);
        FZ_CUDA_CHECK(cudaMalloc(&d_tmpLevel, tmpbLevel));
        cub::DeviceScan::ExclusiveSum(d_tmpBit, tmpbBit, d_len, d_off, nl);
        FZ_CUDA_CHECK(cudaMalloc(&d_tmpBit, tmpbBit));
        FZ_CUDA_CHECK(cudaMalloc(&d_cursor, 4));
    }

    void ensureParseCap(uint64_t nbitsA) {
        if (nbitsA <= cap_nbitsA) return;
        for (void* p : {(void*)d_bflag, (void*)d_brank, (void*)d_ones, (void*)d_gaps})
            if (p) FZ_CUDA_CHECK_WARN(cudaFree(p));
        d_bflag = nullptr; d_brank = nullptr; d_ones = nullptr; d_gaps = nullptr;
        FZ_CUDA_CHECK(cudaMalloc(&d_bflag, nbitsA * 4)); FZ_CUDA_CHECK(cudaMalloc(&d_brank, nbitsA * 4));
        FZ_CUDA_CHECK(cudaMalloc(&d_ones, nbitsA * 8)); FZ_CUDA_CHECK(cudaMalloc(&d_gaps, nbitsA * 4));
        cap_nbitsA = nbitsA;
    }
    void ensureStreamCap(size_t nbytes) {
        if (nbytes <= cap_stream) return;
        if (d_stream) FZ_CUDA_CHECK_WARN(cudaFree(d_stream));
        size_t nb = nbytes ? nbytes : 1;
        FZ_CUDA_CHECK(cudaMalloc(&d_stream, nb));
        cap_stream = nb;
    }
};

Speck2DStage::~Speck2DStage() { delete impl_; }

void Speck2DStage::postStreamSync(cudaStream_t /*stream*/) {
    if (!pending_ || !impl_) return;
    last_B_ = impl_->pend_B;
    last_nbitsA_ = (uint64_t)impl_->pend_offA + (uint64_t)impl_->pend_bitsA;
    last_nbitsB_ = (uint64_t)impl_->pend_offB + (uint64_t)impl_->pend_bitsB;
    uint64_t total_bits = last_nbitsA_ + last_nbitsB_;
    actual_output_size_ = (total_bits + 7) / 8;
    pending_ = false;
}

void Speck2DStage::execute(cudaStream_t stream, MemoryPool* /*pool*/,
                           const std::vector<void*>& inputs,
                           const std::vector<void*>& outputs,
                           const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("Speck2DStage: inputs, outputs, and sizes must be non-empty");
    if (dims_[0] == 0 || dims_[1] == 0)
        throw std::runtime_error("Speck2DStage: dimensions not set — call setDims() before compress/decompress");
    if (dims_[2] > 1)
        throw std::runtime_error("Speck2DStage: 3-D not yet supported (see memory/speck_gpu_design.md P4)");

    const int nx = (int)dims_[0], ny = (int)dims_[1];
    const size_t n = (size_t)nx * ny;

    if (!impl_) impl_ = new Impl();
    impl_->ensureShape(nx, ny);
    Impl& I = *impl_;
    auto g = [&](int cnt) { return dim3((unsigned)((cnt + 255) / 256)); };

    if (!is_inverse_) {
        // ── Forward: int32 codes -> compressed bitstream ───────────────────────
        if (sizes[0] < n * sizeof(int32_t))
            throw std::runtime_error("Speck2DStage: input smaller than dims imply");
        const int32_t* d_code = static_cast<const int32_t*>(inputs[0]);

        k_split_sign_magnitude<<<g((int)n), 256, 0, stream>>>(d_code, (int)n, I.d_mag, I.d_sgn);
        FZ_CUDA_CHECK(cudaGetLastError());

        FZ_CUDA_CHECK(cudaMemsetAsync(I.d_vis, 0, I.nn, stream));
        k_msb<<<g((int)n), 256, 0, stream>>>(I.d_mag, (int)n, I.d_msb);
        for (int L = I.maxL; L >= I.nFused; --L) if (I.level_count[L])
            k_onset<<<g(I.level_count[L]), 256, 0, stream>>>(
                I.d_levelnodes + I.level_start[L], I.level_count[L],
                I.d_lf, I.d_px, I.d_c0, I.d_c1, I.d_c2, I.d_c3, I.d_msb, I.d_on);
        if (I.nFused > 0)
            k_encode_shallow<<<1, kShallowCap, 0, stream>>>(
                I.nFused, I.d_lvl_starts, I.d_lvl_counts, I.d_levelnodes,
                I.d_lf, I.d_px, I.d_c0, I.d_c1, I.d_c2, I.d_c3, I.d_par, I.d_msb, I.d_on, I.d_vis);
        for (int L = I.nFused; L <= I.maxL; ++L) if (I.level_count[L])
            k_visited<<<g(I.level_count[L]), 256, 0, stream>>>(
                I.d_levelnodes + I.level_start[L], I.level_count[L], I.d_par, I.d_on, I.d_vis);

        k_bitsA<<<g(I.nn), 256, 0, stream>>>(I.nn, I.d_levelnodes, I.d_par, I.d_vis, I.d_on, I.d_bitsA);
        cub::DeviceScan::ExclusiveSum(I.d_tmpA, I.tmpbA, I.d_bitsA, I.d_offA, I.nn, stream);
        k_bitsB<<<g(I.nl), 256, 0, stream>>>(I.nl, I.d_leaves, I.d_vis, I.d_on, I.d_bitsB);
        cub::DeviceScan::ExclusiveSum(I.d_tmpB, I.tmpbB, I.d_bitsB, I.d_offB, I.nl, stream);

        FZ_CUDA_CHECK(cudaMemcpyAsync(&I.pend_B, I.d_on, 4, cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(&I.pend_offA, I.d_offA + I.nn - 1, 4, cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(&I.pend_bitsA, I.d_bitsA + I.nn - 1, 4, cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(&I.pend_offB, I.d_offB + I.nl - 1, 4, cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(&I.pend_bitsB, I.d_bitsB + I.nl - 1, 4, cudaMemcpyDeviceToHost, stream));
        pending_ = true;

        // Worst case (see estimateOutputSizes doc): (nn present-slots + nl
        // leaf-slots) words, each <=32 bits. Safe regardless of data.
        size_t words_ub = (size_t)I.nn + (size_t)I.nl + 4;
        if (words_ub > I.out_words_cap) {
            if (I.d_out) FZ_CUDA_CHECK(cudaFree(I.d_out));
            FZ_CUDA_CHECK(cudaMalloc(&I.d_out, words_ub * 4));
            I.out_words_cap = words_ub;
        }
        FZ_CUDA_CHECK(cudaMemsetAsync(I.d_out, 0, I.out_words_cap * 4, stream));
        k_packA<<<g(I.nn), 256, 0, stream>>>(I.nn, I.d_levelnodes, I.d_par, I.d_vis, I.d_on, I.d_offA, I.d_out);

        // packB needs nbitsA as a host value (an absolute bit offset into
        // `out`) -- this is the ONE point this stage needs the pending reads
        // completed early, sacrificing full async-until-postStreamSync purity
        // for a much simpler implementation (isGraphCompatible() is already
        // false for this stage; this sync doesn't cost it anything further).
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
        uint64_t nbitsA = (uint64_t)I.pend_offA + (uint64_t)I.pend_bitsA;
        k_packB<<<g(I.nl), 256, 0, stream>>>(I.nl, I.d_leaves, I.d_vis, I.d_on, I.d_px,
                                             I.d_mag, I.d_sgn, nbitsA, I.d_offB, I.d_out);
        FZ_CUDA_CHECK(cudaGetLastError());

        if (outputs[0] != (void*)I.d_out) {
            // Copy into the pipeline-provided output buffer once the pack
            // kernels above have run (still async on `stream`; caller syncs).
            size_t words = (size_t)((nbitsA + (uint64_t)I.pend_offB + (uint64_t)I.pend_bitsB + 31) / 32);
            if (words == 0) words = 1;
            FZ_CUDA_CHECK(cudaMemcpyAsync(outputs[0], I.d_out, words * 4, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        // ── Inverse: compressed bitstream -> int32 codes ───────────────────────
        const int B = last_B_;
        const uint64_t nbitsA = last_nbitsA_;
        FZ_CUDA_CHECK(cudaMemsetAsync(I.d_decoeff, 0, n * 4, stream));
        FZ_CUDA_CHECK(cudaMemsetAsync(I.d_desgn, 0, n, stream));
        FZ_CUDA_CHECK(cudaMemsetAsync(I.d_present, 0, I.nn, stream));
        if (B >= 0) {
            const size_t nbytes = sizes[0];
            I.ensureStreamCap(nbytes);
            FZ_CUDA_CHECK(cudaMemcpyAsync(I.d_stream, inputs[0], nbytes, cudaMemcpyDeviceToDevice, stream));

            I.ensureParseCap(nbitsA > 0 ? nbitsA : 1);
            if (nbitsA > 0) {
                k_bitflags<<<g((int)nbitsA), 256, 0, stream>>>(I.d_stream, nbitsA, I.d_bflag);
                void* tmp2 = nullptr; size_t tmpb2 = 0;
                cub::DeviceScan::ExclusiveSum(tmp2, tmpb2, I.d_bflag, I.d_brank, (int)nbitsA, stream);
                FZ_CUDA_CHECK(cudaMalloc(&tmp2, tmpb2));
                cub::DeviceScan::ExclusiveSum(tmp2, tmpb2, I.d_bflag, I.d_brank, (int)nbitsA, stream);
                int last_r, last_f;
                FZ_CUDA_CHECK(cudaMemcpyAsync(&last_r, I.d_brank + nbitsA - 1, 4, cudaMemcpyDeviceToHost, stream));
                FZ_CUDA_CHECK(cudaMemcpyAsync(&last_f, I.d_bflag + nbitsA - 1, 4, cudaMemcpyDeviceToHost, stream));
                FZ_CUDA_CHECK(cudaStreamSynchronize(stream));   // need num_ones before scatter/gaps launch shapes
                int num_ones = last_r + last_f;
                k_scatter_ones<<<g((int)nbitsA), 256, 0, stream>>>(I.d_bflag, I.d_brank, nbitsA, I.d_ones);
                k_gaps<<<g(num_ones), 256, 0, stream>>>(I.d_ones, num_ones, I.d_gaps);
                FZ_CUDA_CHECK(cudaFree(tmp2));
            }

            int cursor = 0;
            if (I.nFused > 0) {
                k_decode_shallow<<<1, kShallowCap, 0, stream>>>(
                    I.nFused, I.d_lvl_starts, I.d_lvl_counts, I.d_levelnodes, I.d_par, B,
                    I.d_present, I.d_on, I.d_gaps, I.d_cursor);
                FZ_CUDA_CHECK(cudaMemcpyAsync(&cursor, I.d_cursor, 4, cudaMemcpyDeviceToHost, stream));
                FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            for (int L = I.nFused; L <= I.maxL; ++L) if (I.level_count[L]) {
                int C = I.level_count[L];
                int* nodes = I.d_levelnodes + I.level_start[L];
                k_present<<<g(C), 256, 0, stream>>>(nodes, C, I.d_par, I.d_on, B, I.d_present, I.d_flag);
                cub::DeviceScan::ExclusiveSum(I.d_tmpLevel, I.tmpbLevel, I.d_flag, I.d_rank, C, stream);
                int lr, lf;
                FZ_CUDA_CHECK(cudaMemcpyAsync(&lr, I.d_rank + C - 1, 4, cudaMemcpyDeviceToHost, stream));
                FZ_CUDA_CHECK(cudaMemcpyAsync(&lf, I.d_flag + C - 1, 4, cudaMemcpyDeviceToHost, stream));
                FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
                k_assign<<<g(C), 256, 0, stream>>>(nodes, C, I.d_present, I.d_rank, cursor, I.d_par, I.d_on, B, I.d_gaps);
                cursor += lr + lf;
            }

            k_leaflen<<<g(I.nl), 256, 0, stream>>>(I.d_leaves, I.nl, I.d_present, I.d_on, I.d_len);
            cub::DeviceScan::ExclusiveSum(I.d_tmpBit, I.tmpbBit, I.d_len, I.d_off, I.nl, stream);
            k_fill<<<g(I.nl), 256, 0, stream>>>(I.d_leaves, I.nl, I.d_present, I.d_on, I.d_px,
                                                I.d_stream, nbitsA, I.d_off, I.d_decoeff, I.d_desgn);
        }
        k_join_sign_magnitude<<<g((int)n), 256, 0, stream>>>(
            I.d_decoeff, I.d_desgn, (int)n, static_cast<int32_t*>(outputs[0]));
        FZ_CUDA_CHECK(cudaGetLastError());
        actual_output_size_ = n * sizeof(int32_t);
    }
}

} // namespace fz

// ── FZM-header reconstruction (self-registered; see stage_registry.h) ─────────
FZ_REGISTER_SIMPLE_STAGE(fz::StageType::SPECK2D, fz::Speck2DStage);
