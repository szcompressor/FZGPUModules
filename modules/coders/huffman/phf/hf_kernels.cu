// Adapted from PHF reference (origin/v1.1.0_dev:modules/codec/huffman/hf_kernels.cuhip.inl)
// Changes:
//   - Converted from .inl to a standalone .cu compilation unit (removed include guard).
//   - Removed self-include "#include hf_kernels.cuhip.inl" (self-include present in reference).
//   - Replaced #include "hf_hl.hh" with hf_buf.h (provides HuffmanHelper).
//   - Removed #include "timer.hh".
//   - Added #include "cuda_check.h"; replaced CHECK_GPU → FZ_CUDA_CHECK.
//   - Added (void) casts for unused local variables to silence warnings.
//   - Added explicit template instantiations for the three supported symbol types.
//   - Added GPU_encode_scan (CUB ExclusiveSum), GPU_encode_finalize_totals (custom
//     reduce kernel + async D→H), GPU_cub_scan_temp_bytes, and updated GPU_fine_encode
//     to use them for a fully GPU-async fine encode path (no mid-encode CPU sync).
//   - Removed break handler from KERNEL_CUHIP_Huffman_ReVISIT_lite: it read
//     s_book[MaxBkLen] out-of-bounds (one past the end of the MaxBkLen-element array),
//     aliasing uninitialized s_reduced[0] and corrupting par_ncell.  Fine path is
//     restricted to max_codelen ≤ 8 by hf_hl.cc::encode(), so the shard accumulator
//     (ShardSize=4, BITWIDTH=32) never overflows and the handler is unreachable.
//   - Removed KERNEL_CUHIP_scatter and GPU_scatter: the scatter re-integration step
//     (second half of break handling) was never called from GPU_fine_encode — dead code.
//   - Removed break parameters (brval/bridx/brnum) from kernel, GPU_fine_encode_phase1_2,
//     and GPU_fine_encode.

#include <cstdio>
#include <numeric>
#include <stdexcept>

#include <cub/cub.cuh>

#include "cuda_check.h"
#include "hf_buf.h"
#include "hf_impl.hh"

#define TIX threadIdx.x
#define BIX blockIdx.x
#define BDX blockDim.x

using BYTE = uint8_t;

extern __shared__ char __codec_raw[];

namespace {
struct helper {
    __device__ __forceinline__ static unsigned int local_tid_1()  { return threadIdx.x; }
    __device__ __forceinline__ static unsigned int global_tid_1() { return blockIdx.x * blockDim.x + threadIdx.x; }
    __device__ __forceinline__ static unsigned int block_stride_1() { return blockDim.x; }
    __device__ __forceinline__ static unsigned int grid_stride_1()  { return blockDim.x * gridDim.x; }

    template <int SEQ>
    __device__ __forceinline__ static unsigned int global_tid()   { return blockIdx.x * blockDim.x * SEQ + threadIdx.x; }
    template <int SEQ>
    __device__ __forceinline__ static unsigned int grid_stride()  { return blockDim.x * gridDim.x * SEQ; }
};
}  // namespace

// ── Coarse encode kernels ────────────────────────────────────────────────────

namespace phf {

template <typename E, typename H>
__global__ void KERNEL_CUHIP_encode_phase1_fill(
    E* in, size_t const in_len, H* in_bk, int const in_bklen, H* out_encoded)
{
    auto s_bk = reinterpret_cast<H*>(__codec_raw);

    for (auto idx = helper::local_tid_1(); idx < in_bklen; idx += helper::block_stride_1())
        s_bk[idx] = in_bk[idx];
    __syncthreads();

    for (auto idx = helper::global_tid_1(); idx < in_len; idx += helper::grid_stride_1())
        out_encoded[idx] = s_bk[(int)in[idx]];
}

template <typename H, typename M>
__global__ void KERNEL_CUHIP_encode_phase2_deflate(
    H* inout_inplace, size_t const len, M* par_nbit, M* par_ncell,
    int const sublen, int const pardeg)
{
    constexpr int CELL_BITWIDTH = sizeof(H) * 8;
    auto tid = BIX * BDX + TIX;

    if (tid * sublen < len) {
        int residue_bits = CELL_BITWIDTH;
        int total_bits   = 0;
        H*  ptr          = inout_inplace + tid * sublen;
        H   bufr;
        uint8_t word_width;

        auto did = tid * sublen;
        for (auto i = 0; i < sublen; i++, did++) {
            if (did == len) break;

            H   packed_word = inout_inplace[tid * sublen + i];
            auto word_ptr   = reinterpret_cast<struct HuffmanWord<sizeof(H)>*>(&packed_word);
            word_width           = word_ptr->bitcount;
            word_ptr->bitcount   = (uint8_t)0x0;

            if (residue_bits == CELL_BITWIDTH) bufr = 0x0;

            if (word_width <= residue_bits) {
                residue_bits -= word_width;
                bufr |= packed_word << residue_bits;
                if (residue_bits == 0) {
                    residue_bits = CELL_BITWIDTH;
                    *(ptr++) = bufr;
                }
            }
            else {
                auto l_bits = word_width - residue_bits;
                auto r_bits = CELL_BITWIDTH - l_bits;
                bufr |= packed_word >> l_bits;
                *(ptr++) = bufr;
                bufr         = packed_word << r_bits;
                residue_bits = r_bits;
            }
            total_bits += word_width;
        }
        *ptr = bufr;

        par_nbit[tid]  = total_bits;
        par_ncell[tid] = (total_bits + CELL_BITWIDTH - 1) / CELL_BITWIDTH;
    }
}

template <typename H, typename M>
__global__ void KERNEL_CUHIP_encode_phase4_concatenate(
    H* gapped, M* par_entry, M* par_ncell, int const cfg_sublen, H* non_gapped)
{
    auto n   = par_ncell[blockIdx.x];
    auto src = gapped + cfg_sublen * blockIdx.x;
    auto dst = non_gapped + par_entry[blockIdx.x];
    for (auto i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

}  // namespace phf

// ── ReVISIT-lite (fine encode) kernel ────────────────────────────────────────
// Requires max Huffman code length ≤ BITWIDTH/ShardSize (8 bits for the default
// ShardSize=4, BITWIDTH=32).  hf_hl.cc::encode() enforces this before choosing
// the fine path so the shard accumulator never overflows.

namespace phf {

using Hf = uint32_t;

template <typename E, int ChunkSize = 1024, int ShardSize = 4, int MaxBkLen = 1024>
__global__ void KERNEL_CUHIP_Huffman_ReVISIT_lite(
    E* in_data, size_t const len, Hf* hf_book, const uint32_t runtime_bklen,
    uint32_t* hf_bitstream, uint32_t* hf_bits, uint32_t* hf_cells,
    const uint32_t nblock)
{
    constexpr auto NumThreads = ChunkSize / ShardSize;

    __shared__ E  s_to_encode[ChunkSize];
    auto const id_base = blockIdx.x * ChunkSize;

#pragma unroll
    for (auto ix = 0; ix < ShardSize; ix++) {
        auto id = id_base + threadIdx.x + ix * NumThreads;
        if (id < len) s_to_encode[threadIdx.x + ix * NumThreads] = in_data[id];
    }
    __syncthreads();

    constexpr auto ReduceTimes  = 2u;
    constexpr auto ShuffleTimes = 8u;
    constexpr auto BITWIDTH     = 32;

    static_assert(ShardSize  == 1 << ReduceTimes,                   "Wrong reduce times.");
    static_assert(ChunkSize  == 1 << (ReduceTimes + ShuffleTimes),  "Wrong shuffle times.");

    __shared__ Hf       s_book[MaxBkLen];
    __shared__ Hf       s_reduced[NumThreads * 2];
    __shared__ uint32_t s_bitcount[NumThreads * 2];

    auto bitcount_of = [](Hf* _w) { return reinterpret_cast<HuffmanWord<4>*>(_w)->bitcount; };
    auto entry       = [&]() -> size_t { return ChunkSize * blockIdx.x; };
    auto allowed_len = [&]() { return min((size_t)ChunkSize, len - entry()); };

    for (auto i = threadIdx.x; i < runtime_bklen; i += NumThreads) s_book[i] = hf_book[i];
    __syncthreads();

    // reduce-merge
    {
        auto p_bits{0u};
        Hf   p_reduced{0x0};

        for (auto i = 0; i < ShardSize; i++) {
            auto idx      = (threadIdx.x * ShardSize) + i;
            auto p_key    = s_to_encode[idx];
            auto p_val    = s_book[p_key];
            auto sym_bits = bitcount_of(&p_val);

            p_val <<= (BITWIDTH - sym_bits);
            p_reduced |= (p_val >> p_bits);
            p_bits    += sym_bits * (idx < allowed_len());
        }

        s_reduced[threadIdx.x]  = p_reduced;
        s_bitcount[threadIdx.x] = p_bits;
    }
    __syncthreads();

    // shuffle-merge
    for (auto sf = ShuffleTimes, stride = 1u; sf > 0; sf--, stride *= 2) {
        auto l = threadIdx.x / (stride * 2) * (stride * 2);
        auto r = l + stride;

        auto lbc       = s_bitcount[l];
        uint32_t used__units = lbc / BITWIDTH;
        uint32_t used___bits = lbc % BITWIDTH;
        uint32_t unused_bits = BITWIDTH - used___bits;

        auto lend       = (Hf*)(s_reduced + l + used__units);
        auto this_point = s_reduced[threadIdx.x];
        auto lsym       = this_point >> used___bits;
        auto rsym       = this_point << unused_bits;

        if (threadIdx.x >= r && threadIdx.x < r + stride)
            atomicAnd((Hf*)(s_reduced + threadIdx.x), 0x0);
        __syncthreads();

        if (threadIdx.x >= r && threadIdx.x < r + stride) {
            atomicOr(lend + (threadIdx.x - r) + 0, lsym);
            atomicOr(lend + (threadIdx.x - r) + 1, rsym);
        }
        if (threadIdx.x == l) s_bitcount[l] += s_bitcount[r];
        __syncthreads();
    }

    // output
    __shared__ uint32_t s_wunits;
    unsigned long long  p_wunits;

    static_assert(BITWIDTH == 32, "Wrong bitwidth (!=32).");
    if (threadIdx.x == 0) {
        uint32_t p_bc = s_bitcount[0];
        p_wunits = (p_bc + 31) / 32;
        hf_bits[blockIdx.x]  = p_bc;
        hf_cells[blockIdx.x] = p_wunits;
        s_wunits = p_wunits;
    }
    __syncthreads();

    if (threadIdx.x % 32 == 0 && threadIdx.x / 32 > 0) p_wunits = s_wunits;
    __syncthreads();

    p_wunits = __shfl_sync(0xffffffff, p_wunits, 0);

    for (auto i = threadIdx.x; i < p_wunits; i += blockDim.x)
        hf_bitstream[id_base + i] = s_reduced[i];
}

}  // namespace phf

// ── Decode kernel ─────────────────────────────────────────────────────────────

namespace phf {

template <typename E, typename H, typename M>
__global__ void KERNEL_CUHIP_HF_decode(
    H* in, uint8_t* revbook, M* par_nbit, M* par_entry,
    int const revbook_nbyte, int const sublen, int const pardeg, E* out)
{
    constexpr auto CELL_BITWIDTH = sizeof(H) * 8;
    extern __shared__ uint8_t s_revbook[];
    constexpr auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;

    auto single_thread_inflate = [&](H* input, E* out, int const total_bw) {
        int next_bit;
        auto idx_bit = 0, idx_byte = 0, idx_out = 0;
        H    bufr    = input[idx_byte];
        auto first   = (H*)(s_revbook);
        auto entry   = first + CELL_BITWIDTH;
        auto keys    = (E*)(s_revbook + sizeof(H) * (2 * CELL_BITWIDTH));
        H    v       = (bufr >> (CELL_BITWIDTH - 1)) & 0x1;
        auto l       = 1, i = 0;

        while (i < total_bw) {
            while (v < first[l]) {
                ++i;
                idx_byte = i / CELL_BITWIDTH;
                idx_bit  = i % CELL_BITWIDTH;
                if (idx_bit == 0) bufr = input[idx_byte];
                next_bit = ((bufr >> (CELL_BITWIDTH - 1 - idx_bit)) & 0x1);
                v = (v << 1) | next_bit;
                ++l;
            }
            out[idx_out++] = keys[entry[l] + v - first[l]];
            {
                ++i;
                idx_byte = i / CELL_BITWIDTH;
                idx_bit  = i % CELL_BITWIDTH;
                if (idx_bit == 0) bufr = input[idx_byte];
                next_bit = ((bufr >> (CELL_BITWIDTH - 1 - idx_bit)) & 0x1);
                v = 0x0 | next_bit;
            }
            l = 1;
        }
    };

    auto R = (revbook_nbyte - 1 + block_dim) / block_dim;
    for (auto i = 0; i < R; i++) {
        if (TIX + i * block_dim < revbook_nbyte)
            s_revbook[TIX + i * block_dim] = revbook[TIX + i * block_dim];
    }
    __syncthreads();

    auto gid = BIX * BDX + TIX;
    if (gid < pardeg) {
        single_thread_inflate(in + par_entry[gid], out + sublen * gid, par_nbit[gid]);
        __syncthreads();
    }
}

}  // namespace phf

// ── Fine-path: combined nbit+ncell reduction kernel ───────────────────────────
// Single-block grid-stride reduction. Accumulates in uint64_t to avoid overflow
// for inputs where len * bits_per_symbol > 2^32.

namespace phf {

template <typename M>
__global__ void KERNEL_phase3_reduce(
    const M* par_nbit, const M* par_ncell, int pardeg,
    uint64_t* total_nbit, uint64_t* total_ncell)
{
    extern __shared__ uint64_t smem[];
    uint64_t* s_nbit  = smem;
    uint64_t* s_ncell = smem + blockDim.x;

    uint64_t nbit = 0, ncell = 0;
    for (int i = threadIdx.x; i < pardeg; i += blockDim.x) {
        nbit  += par_nbit[i];
        ncell += par_ncell[i];
    }
    s_nbit[threadIdx.x]  = nbit;
    s_ncell[threadIdx.x] = ncell;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_nbit[threadIdx.x]  += s_nbit[threadIdx.x + s];
            s_ncell[threadIdx.x] += s_ncell[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *total_nbit  = s_nbit[0];
        *total_ncell = s_ncell[0];
    }
}

}  // namespace phf

// ── phf::cuhip::modules<E,H> method definitions ──────────────────────────────

#define PHF_MODULE_TPL   template <typename E, typename H>
#define PHF_MODULE_CLASS phf::cuhip::modules<E, H>
#define SETUP_DIV                                                    \
    auto div = [](auto whole, auto part) -> uint32_t {               \
        if (whole == 0) throw std::runtime_error("Dividend is zero."); \
        if (part  == 0) throw std::runtime_error("Divisor is zero.");  \
        return (whole - 1) / part + 1;                               \
    };

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode_phase1(
    E* in_data, const size_t data_len, H* in_book, const uint32_t book_len,
    const int numSMs, H* out_bitstream, void* stream)
{
    SETUP_DIV;
    constexpr auto block_dim = HuffmanHelper::BLOCK_DIM_ENCODE;
    (void)div; (void)block_dim; // grid_dim not used; kernel launched with 8*numSMs
    phf::KERNEL_CUHIP_encode_phase1_fill<E, H>
        <<<8 * numSMs, 256, sizeof(H) * book_len, (cudaStream_t)stream>>>
        (in_data, data_len, in_book, book_len, out_bitstream);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode_phase2(
    H* in_data, const size_t data_len, phf::par_config hfpar, H* deflated,
    M* par_nbit, M* par_ncell, void* stream)
{
    SETUP_DIV;
    constexpr auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;
    auto grid_dim = div(hfpar.pardeg, block_dim);
    phf::KERNEL_CUHIP_encode_phase2_deflate<H>
        <<<grid_dim, block_dim, 0, (cudaStream_t)stream>>>
        (deflated, data_len, par_nbit, par_ncell, hfpar.sublen, hfpar.pardeg);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_fine_encode_phase1_2(
    E* in, const size_t len, H* book, const uint32_t bklen, H* bitstream,
    M* par_nbit, M* par_ncell, const uint32_t nblock, void* stream)
{
    SETUP_DIV;
    constexpr int ChunkSize = 1024;
    constexpr int BlockDim  = 256;
    auto grid_dim = div(len, ChunkSize);
    phf::KERNEL_CUHIP_Huffman_ReVISIT_lite<E>
        <<<grid_dim, BlockDim, 0, (cudaStream_t)stream>>>
        (in, len, book, bklen, bitstream, par_nbit, par_ncell, nblock);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode_phase3_sync(
    phf::par_config hfpar, M* d_par_nbit, M* h_par_nbit,
    M* d_par_ncell, M* h_par_ncell, M* d_par_entry, M* h_par_entry,
    size_t* outlen_nbit, size_t* outlen_ncell, float* time_cpu_time, void* stream)
{
    (void)time_cpu_time;

    FZ_CUDA_CHECK(cudaMemcpyAsync(
        h_par_nbit, d_par_nbit, hfpar.pardeg * sizeof(M),
        cudaMemcpyDeviceToHost, (cudaStream_t)stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(
        h_par_ncell, d_par_ncell, hfpar.pardeg * sizeof(M),
        cudaMemcpyDeviceToHost, (cudaStream_t)stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));

    memcpy(h_par_entry + 1, h_par_ncell, (hfpar.pardeg - 1) * sizeof(M));
    for (auto i = 1; i < hfpar.pardeg; i++) h_par_entry[i] += h_par_entry[i - 1];

    if (outlen_nbit)
        *outlen_nbit  = std::accumulate(h_par_nbit,  h_par_nbit  + hfpar.pardeg, (size_t)0);
    if (outlen_ncell)
        *outlen_ncell = std::accumulate(h_par_ncell, h_par_ncell + hfpar.pardeg, (size_t)0);

    FZ_CUDA_CHECK(cudaMemcpyAsync(
        d_par_entry, h_par_entry, hfpar.pardeg * sizeof(M),
        cudaMemcpyHostToDevice, (cudaStream_t)stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode_phase4(
    H* in_buf, const size_t /*len*/, M* par_entry, M* par_ncell,
    phf::par_config hfpar, H* bitstream, const size_t /*max_bitstream_len*/, void* stream)
{
    phf::KERNEL_CUHIP_encode_phase4_concatenate<H, M>
        <<<hfpar.pardeg, 128, 0, (cudaStream_t)stream>>>
        (in_buf, par_entry, par_ncell, hfpar.sublen, bitstream);
}

PHF_MODULE_TPL size_t PHF_MODULE_CLASS::GPU_cub_scan_temp_bytes(size_t pardeg)
{
    size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, (M*)nullptr, (M*)nullptr, (int)pardeg);
    return bytes;
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_encode_scan(
    M* d_par_ncell, M* d_par_entry, int pardeg,
    uint8_t* d_cub_temp, size_t cub_temp_bytes, void* stream)
{
    cub::DeviceScan::ExclusiveSum(
        d_cub_temp, cub_temp_bytes, d_par_ncell, d_par_entry, pardeg,
        (cudaStream_t)stream);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_encode_finalize_totals(
    M* d_par_nbit, M* d_par_ncell, int pardeg,
    uint64_t* d_total_nbit, uint64_t* d_total_ncell,
    uint64_t* h_total_nbit, uint64_t* h_total_ncell,
    void* stream)
{
    auto s = (cudaStream_t)stream;
    constexpr int block = 256;
    phf::KERNEL_phase3_reduce<M>
        <<<1, block, 2 * block * sizeof(uint64_t), s>>>
        (d_par_nbit, d_par_ncell, pardeg, d_total_nbit, d_total_ncell);

    FZ_CUDA_CHECK(cudaMemcpyAsync(
        h_total_nbit,  d_total_nbit,  sizeof(uint64_t), cudaMemcpyDeviceToHost, s));
    FZ_CUDA_CHECK(cudaMemcpyAsync(
        h_total_ncell, d_total_ncell, sizeof(uint64_t), cudaMemcpyDeviceToHost, s));
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode(
    E* in_data, size_t data_len, H* in_book, uint32_t book_len, int numSMs,
    phf::par_config hfpar,
    H* d_scratch4, M* d_par_nbit, M* h_par_nbit,
    M* d_par_ncell, M* h_par_ncell, M* d_par_entry, M* h_par_entry,
    H* d_bitstream4, size_t bitstream_max_len,
    size_t* out_total_nbit, size_t* out_total_ncell, void* stream)
{
    GPU_coarse_encode_phase1(in_data, data_len, in_book, book_len, numSMs, d_scratch4, stream);
    GPU_coarse_encode_phase2(d_scratch4, data_len, hfpar, d_scratch4, d_par_nbit, d_par_ncell, stream);
    GPU_coarse_encode_phase3_sync(
        hfpar, d_par_nbit, h_par_nbit, d_par_ncell, h_par_ncell, d_par_entry, h_par_entry,
        out_total_nbit, out_total_ncell, nullptr, stream);
    GPU_coarse_encode_phase4(d_scratch4, data_len, d_par_entry, d_par_ncell, hfpar, d_bitstream4, bitstream_max_len, stream);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_fine_encode(
    E* in_data, size_t data_len, H* in_book, uint32_t book_len,
    phf::par_config hfpar,
    H* d_scratch4,
    M* d_par_nbit, M* d_par_ncell, M* d_par_entry,
    H* d_bitstream4, size_t bitstream_max_len,
    uint64_t* d_total_nbit, uint64_t* d_total_ncell,
    uint64_t* h_total_nbit, uint64_t* h_total_ncell,
    uint8_t* d_cub_temp, size_t cub_temp_bytes, void* stream)
{
    // Phase 1+2: ReVISIT-lite single kernel (encode + reduce-merge in shared mem)
    GPU_fine_encode_phase1_2(
        in_data, data_len, in_book, book_len, d_scratch4, d_par_nbit, d_par_ncell,
        hfpar.pardeg, stream);

    // GPU-async scan: d_par_ncell → d_par_entry (partition bitstream offsets)
    GPU_encode_scan(d_par_ncell, d_par_entry, (int)hfpar.pardeg,
                    d_cub_temp, cub_temp_bytes, stream);

    // Phase 4: scatter partitions to final bitstream positions using d_par_entry
    GPU_coarse_encode_phase4(
        d_scratch4, data_len, d_par_entry, d_par_ncell, hfpar, d_bitstream4,
        bitstream_max_len, stream);

    // Reduce total nbit+ncell and async-copy to pinned mem; ready after caller's sync
    GPU_encode_finalize_totals(
        d_par_nbit, d_par_ncell, (int)hfpar.pardeg,
        d_total_nbit, d_total_ncell, h_total_nbit, h_total_ncell, stream);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_decode(
    H* in_bitstream, uint8_t* in_revbook, size_t const revbook_len,
    M* in_par_nbit, M* in_par_entry, size_t const sublen, size_t const pardeg,
    E* out_decoded, void* stream)
{
    SETUP_DIV;
    auto const block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;
    auto const grid_dim  = div(pardeg, block_dim);
    phf::KERNEL_CUHIP_HF_decode<E, H, M>
        <<<grid_dim, block_dim, revbook_len, (cudaStream_t)stream>>>
        (in_bitstream, in_revbook, in_par_nbit, in_par_entry, revbook_len, sublen, pardeg, out_decoded);
}

#undef PHF_MODULE_TPL
#undef PHF_MODULE_CLASS
#undef SETUP_DIV
#undef TIX
#undef BIX
#undef BDX

// ── Explicit instantiations ───────────────────────────────────────────────────

template class phf::cuhip::modules<uint8_t,  uint32_t>;
template class phf::cuhip::modules<uint16_t, uint32_t>;
template class phf::cuhip::modules<uint32_t, uint32_t>;
