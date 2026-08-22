// Adapted from PHF reference (origin/v1.1.0_dev:modules/codec/huffman/hf_kernels.cuhip.inl)
// Changes:
//   - Converted from .inl to a standalone .cu compilation unit (removed include guard).
//   - Removed self-include "#include hf_kernels.cuhip.inl" (self-include present in reference).
//   - Replaced #include "hf_hl.hh" with hf_buf.h (provides HuffmanHelper).
//   - Removed #include "timer.hh".
//   - Added #include "cuda_check.h"; replaced CHECK_GPU → FZ_CUDA_CHECK.
//   - Added (void) casts for unused local variables to silence warnings.
//   - Guarded post-symbol bit fetch in KERNEL_CUHIP_HF_decode's
//     single_thread_inflate (line ~298): when total_bw is a multiple of 32 the
//     speculative input[idx_byte] read after the last emitted symbol fell one
//     word past the partition slice (and one word past the entire encoded
//     bitstream allocation on the final partition). Now skipped when
//     i >= total_bw — the outer while loop catches that on the next iteration.
//     Surfaced by compute-sanitizer as a 4-byte OOB read at address+1B-past-end.

#include <algorithm>
#include <numeric>
#include <stdexcept>

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
                // Skip the bit fetch when we've consumed the last bit of this
                // partition: the outer `while (i < total_bw)` exit will catch
                // it on the next iteration. Without this guard, if total_bw is
                // an exact multiple of CELL_BITWIDTH the speculative
                // `input[idx_byte]` reads one word past the partition slice
                // (and past the encoded bitstream allocation on the final
                // partition) — caught by compute-sanitizer as an OOB read of
                // size 4 at the address-after-end.
                if (i < total_bw) {
                    idx_byte = i / CELL_BITWIDTH;
                    idx_bit  = i % CELL_BITWIDTH;
                    if (idx_bit == 0) bufr = input[idx_byte];
                    next_bit = ((bufr >> (CELL_BITWIDTH - 1 - idx_bit)) & 0x1);
                    v = 0x0 | next_bit;
                }
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

template <typename E, typename H, typename M>
__global__ void KERNEL_CUHIP_HF_decode_device_header(
    PHF_BYTE* encoded, int revbook_nbyte, E* out)
{
    constexpr int CELL_BITWIDTH = sizeof(H) * 8;
    extern __shared__ uint8_t s_revbook[];

    const auto* header = reinterpret_cast<const phf_header*>(encoded);
    const auto* revbook = encoded + header->entry[PHFHEADER_REVBK];
    auto* bitstream = reinterpret_cast<H*>(
        encoded + header->entry[PHFHEADER_BITSTREAM]);
    auto* par_nbit = reinterpret_cast<M*>(
        encoded + header->entry[PHFHEADER_PAR_NBIT]);
    auto* par_entry = reinterpret_cast<M*>(
        encoded + header->entry[PHFHEADER_PAR_ENTRY]);

    for (int i = threadIdx.x; i < revbook_nbyte; i += blockDim.x)
        s_revbook[i] = revbook[i];
    __syncthreads();

    auto decode_partition = [&](H* input, E* partition_out, int total_bw) {
        int next_bit;
        int idx_bit = 0, idx_byte = 0, idx_out = 0;
        H bufr = input[idx_byte];
        auto first = reinterpret_cast<H*>(s_revbook);
        auto entry = first + CELL_BITWIDTH;
        auto keys = reinterpret_cast<E*>(s_revbook + sizeof(H) * (2 * CELL_BITWIDTH));
        H v = (bufr >> (CELL_BITWIDTH - 1)) & 0x1;
        int l = 1, i = 0;

        while (i < total_bw) {
            while (v < first[l]) {
                ++i;
                idx_byte = i / CELL_BITWIDTH;
                idx_bit = i % CELL_BITWIDTH;
                if (idx_bit == 0) bufr = input[idx_byte];
                next_bit = (bufr >> (CELL_BITWIDTH - 1 - idx_bit)) & 0x1;
                v = (v << 1) | next_bit;
                ++l;
            }
            partition_out[idx_out++] = keys[entry[l] + v - first[l]];
            ++i;
            if (i < total_bw) {
                idx_byte = i / CELL_BITWIDTH;
                idx_bit = i % CELL_BITWIDTH;
                if (idx_bit == 0) bufr = input[idx_byte];
                next_bit = (bufr >> (CELL_BITWIDTH - 1 - idx_bit)) & 0x1;
                v = next_bit;
            }
            l = 1;
        }
    };

    const int stride = blockDim.x * gridDim.x;
    for (int gid = blockIdx.x * blockDim.x + threadIdx.x;
         gid < header->pardeg; gid += stride) {
        decode_partition(
            bitstream + par_entry[gid], out + static_cast<size_t>(header->sublen) * gid,
            static_cast<int>(par_nbit[gid]));
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

    // h_par_entry is an EXCLUSIVE prefix sum, so entry 0 is 0 by definition — but
    // nothing wrote it. The memcpy below fills [1, pardeg) and the accumulation
    // starts at i=1, so entry 0 was whatever the buffer already held.
    //
    // That is not hypothetical: h_par_entry is pool-allocated pinned memory, so a
    // second HuffmanStage sharing a pool with a first one inherits its bytes, and
    // the stale value then propagates through every entry (each is += its
    // predecessor). A first use of a fresh pool happens to get zeroed memory,
    // which is why this only bites on pool reuse.
    h_par_entry[0] = 0;
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
    phf::par_config hfpar, H* bitstream, const size_t max_bitstream_len,
    const size_t total_ncell, void* stream)
{
    // max_bitstream_len was passed in and ignored by this function. On the coarse
    // path phase3_sync has already brought the exact cell total to the host, so
    // checking it costs nothing and is the last point at which an overflow is a
    // diagnosable error rather than a silent out-of-bounds write into whatever the
    // pool placed after the bitstream.
    //
    // Buf now sizes the bitstream for the worst case the codeword format allows
    // (27 bits/symbol), so this should be unreachable; it stays as the backstop the
    // old `inlen / 2` sizing never had.
    if (total_ncell > max_bitstream_len)
        throw std::runtime_error(
            "cuSZ Huffman encode: concatenated bitstream needs " + std::to_string(total_ncell) +
            " cells but only " + std::to_string(max_bitstream_len) +
            " were allocated. The codebook is a poor enough fit for this data that "
            "the encoding expands past the buffer bound.");

    GPU_coarse_encode_phase4_device(
        in_buf, par_entry, par_ncell, hfpar, bitstream, stream);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode_phase4_device(
    H* in_buf, M* par_entry, M* par_ncell, phf::par_config hfpar,
    H* bitstream, void* stream)
{
    phf::KERNEL_CUHIP_encode_phase4_concatenate<H, M>
        <<<hfpar.pardeg, 128, 0, (cudaStream_t)stream>>>
        (in_buf, par_entry, par_ncell, hfpar.sublen, bitstream);
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
    GPU_coarse_encode_phase4(
        d_scratch4, data_len, d_par_entry, d_par_ncell, hfpar, d_bitstream4,
        bitstream_max_len, out_total_ncell ? *out_total_ncell : 0, stream);
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

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_decode_device(
    PHF_BYTE* in_encoded, size_t revbook_len, size_t estimated_pardeg,
    int numSMs,
    E* out_decoded, void* stream)
{
    constexpr int block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;
    const size_t blocks_for_estimate =
        (estimated_pardeg + block_dim - 1) / block_dim;
    const size_t occupancy_cap = numSMs > 0 ? static_cast<size_t>(8 * numSMs) : 1;
    // The embedded header remains authoritative inside the kernel. This estimate
    // only avoids launching hundreds of idle CTAs for small archives; a smaller
    // grid still covers a larger/older embedded pardeg through the grid-stride loop.
    const int grid_dim = static_cast<int>(
        std::max<size_t>(1, std::min(blocks_for_estimate, occupancy_cap)));
    phf::KERNEL_CUHIP_HF_decode_device_header<E, H, M>
        <<<grid_dim, block_dim, revbook_len, (cudaStream_t)stream>>>
        (in_encoded, static_cast<int>(revbook_len), out_decoded);
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
