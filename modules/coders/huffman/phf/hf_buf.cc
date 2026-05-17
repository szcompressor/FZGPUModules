// Adapted from PHF reference (origin/v1.1.0_dev:modules/codec/huffman/hf_buf.cc)
// Changes:
//   - Replaced #include "hf_hl.hh" with local hf_buf.h.
//   - Replaced #include "err.hh" with cuda_check.h; replaced CHECK_GPU → FZ_CUDA_CHECK.
//   - All device/host allocations now routed through MemoryPool::allocatePersistentDevice
//     / allocatePersistentPinned instead of cudaMalloc / cudaMallocHost directly.
//   - Destructor returns all allocations to the pool via freePersistentDevice /
//     freePersistentPinned.  Pool is the sole owner of all memory.
//   - Removed MAKE_UNIQUE_DEVICE / MAKE_UNIQUE_HOST macros (no longer needed).
//   - Removed .get() calls throughout (members are now raw pointers).
//   - When use_HFR=true, allocates fine-path async-total buffers (d/h_total_nbit/ncell)
//     and CUB temp storage (d_cub_temp); all null otherwise.

#include <cstddef>

#include "hf.h"
#include "hf_buf.h"
#include "cuda_check.h"

namespace phf {

template <typename E>
int Buf<E>::_revbk4_bytes(int bklen)
{
    return phf_reverse_book_bytes(bklen, 4, sizeof(SYM));
}

template <typename E>
int Buf<E>::_revbk8_bytes(int bklen)
{
    return phf_reverse_book_bytes(bklen, 8, sizeof(SYM));
}

// Helper macro for allocatePersistentDevice / allocatePersistentPinned
// to reduce repetition.  Casts the void* return to the correct pointer type.
#define PALLOC_DEV(T, n, tag)  \
    static_cast<T*>(pool->allocatePersistentDevice( \
        static_cast<size_t>(n) * sizeof(T), (tag)))

#define PALLOC_PIN(T, n, tag)  \
    static_cast<T*>(pool->allocatePersistentPinned( \
        static_cast<size_t>(n) * sizeof(T), (tag)))

template <typename E>
Buf<E>::Buf(size_t inlen, size_t _bklen, fz::MemoryPool* pool,
            int _pardeg, bool _use_HFR)
    : len(inlen),
      pardeg((inlen - 1) / ((_use_HFR ? 1024 : capi_phf_coarse_tune_sublen(inlen))) + 1),
      sublen(_use_HFR ? 1024 : capi_phf_coarse_tune_sublen(inlen)),
      bklen(_bklen),
      use_HFR(_use_HFR),
      revbk4_bytes(_revbk4_bytes(_bklen)),
      bitstream_max_len(inlen / 2),
      rt_bklen(0),
      numSMs(0),
      pool_(pool)
{
    (void)_pardeg;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    h_scratch4   = PALLOC_PIN(H4,       len,                "huf_h_scratch4");
    d_scratch4   = PALLOC_DEV(H4,       len,                "huf_d_scratch4");
    h_bk4        = PALLOC_PIN(H4,       bklen,              "huf_h_bk4");
    d_bk4        = PALLOC_DEV(H4,       bklen,              "huf_d_bk4");
    h_revbk4     = PALLOC_PIN(PHF_BYTE, revbk4_bytes,       "huf_h_revbk4");
    d_revbk4     = PALLOC_DEV(PHF_BYTE, revbk4_bytes,       "huf_d_revbk4");
    d_bitstream4 = PALLOC_DEV(H4,       bitstream_max_len,  "huf_d_bitstream4");
    h_bitstream4 = PALLOC_PIN(H4,       bitstream_max_len,  "huf_h_bitstream4");
    h_par_nbit   = PALLOC_PIN(M,        pardeg,             "huf_h_par_nbit");
    d_par_nbit   = PALLOC_DEV(M,        pardeg,             "huf_d_par_nbit");
    h_par_ncell  = PALLOC_PIN(M,        pardeg,             "huf_h_par_ncell");
    d_par_ncell  = PALLOC_DEV(M,        pardeg,             "huf_d_par_ncell");
    h_par_entry  = PALLOC_PIN(M,        pardeg,             "huf_h_par_entry");
    d_par_entry  = PALLOC_DEV(M,        pardeg,             "huf_d_par_entry");

    // Histogram buffers
    d_freq = PALLOC_DEV(uint32_t, bklen, "huf_d_freq");
    h_freq = PALLOC_PIN(uint32_t, bklen, "huf_h_freq");

    // Fine-path async totals and CUB scan temp — only when use_HFR is active
    if (_use_HFR) {
        cub_temp_bytes = phf::cuhip::modules<E, H4>::GPU_cub_scan_temp_bytes(pardeg);
        d_cub_temp    = PALLOC_DEV(uint8_t,  cub_temp_bytes, "huf_d_cub_temp");
        d_total_nbit  = PALLOC_DEV(uint64_t, 1,              "huf_d_total_nbit");
        d_total_ncell = PALLOC_DEV(uint64_t, 1,              "huf_d_total_ncell");
        h_total_nbit  = PALLOC_PIN(uint64_t, 1,              "huf_h_total_nbit");
        h_total_ncell = PALLOC_PIN(uint64_t, 1,              "huf_h_total_ncell");
    } else {
        cub_temp_bytes = 0;
        d_cub_temp    = nullptr;
        d_total_nbit  = nullptr;
        d_total_ncell = nullptr;
        h_total_nbit  = nullptr;
        h_total_ncell = nullptr;
    }

    // d_encoded / h_encoded alias the scratch buffers (not separate allocations)
    d_encoded = reinterpret_cast<PHF_BYTE*>(d_scratch4);
    h_encoded = reinterpret_cast<PHF_BYTE*>(h_scratch4);

    size_t fine_d_extra = _use_HFR ? (cub_temp_bytes + 2 * sizeof(uint64_t)) : 0;
    size_t fine_h_extra = _use_HFR ? (2 * sizeof(uint64_t)) : 0;
    total_footprint_d  = (sizeof(H4) * len) + (sizeof(H4) * bklen) +
                         (sizeof(PHF_BYTE) * revbk4_bytes) +
                         (sizeof(H4) * bitstream_max_len) +
                         (sizeof(M) * pardeg * 3) +
                         (sizeof(uint32_t) * bklen) +
                         fine_d_extra;
    total_footprint_h  = (sizeof(H4) * len) + (sizeof(H4) * bklen) +
                         (sizeof(PHF_BYTE) * revbk4_bytes) +
                         (sizeof(H4) * bitstream_max_len) +
                         (sizeof(M) * pardeg * 3) +
                         (sizeof(uint32_t) * bklen) +
                         fine_h_extra;
}

#undef PALLOC_DEV
#undef PALLOC_PIN

template <typename E>
Buf<E>::~Buf()
{
    // Return all allocations to the pool.  The pool is guaranteed to outlive
    // Buf<E> when used inside a Pipeline (mem_pool_ declared before stages_,
    // destroyed after them in Pipeline's destructor).
    pool_->freePersistentDevice(d_scratch4);
    pool_->freePersistentDevice(d_bk4);
    pool_->freePersistentDevice(d_revbk4);
    pool_->freePersistentDevice(d_bitstream4);
    pool_->freePersistentDevice(d_par_nbit);
    pool_->freePersistentDevice(d_par_ncell);
    pool_->freePersistentDevice(d_par_entry);
    pool_->freePersistentDevice(d_freq);
    if (d_cub_temp)    pool_->freePersistentDevice(d_cub_temp);
    if (d_total_nbit)  pool_->freePersistentDevice(d_total_nbit);
    if (d_total_ncell) pool_->freePersistentDevice(d_total_ncell);

    pool_->freePersistentPinned(h_scratch4);
    pool_->freePersistentPinned(h_bk4);
    pool_->freePersistentPinned(h_revbk4);
    pool_->freePersistentPinned(h_bitstream4);
    pool_->freePersistentPinned(h_par_nbit);
    pool_->freePersistentPinned(h_par_ncell);
    pool_->freePersistentPinned(h_par_entry);
    pool_->freePersistentPinned(h_freq);
    if (h_total_nbit)  pool_->freePersistentPinned(h_total_nbit);
    if (h_total_ncell) pool_->freePersistentPinned(h_total_ncell);
    // d_encoded / h_encoded are aliases — do not free separately
}

template <typename E>
void Buf<E>::memcpy_merge(Header& header, phf_stream_t stream)
{
    auto memcpy_start           = d_encoded;
    auto memcpy_adjust_to_start = 0;

    memcpy_helper _revbk{     d_revbk4,    revbk4_bytes,                   header.entry[PHFHEADER_REVBK]};
    memcpy_helper _par_nbit{  d_par_nbit,  pardeg * sizeof(M),             header.entry[PHFHEADER_PAR_NBIT]};
    memcpy_helper _par_entry{ d_par_entry, pardeg * sizeof(M),             header.entry[PHFHEADER_PAR_ENTRY]};
    memcpy_helper _bitstream{ d_bitstream4, bitstream_max_len * sizeof(H4), header.entry[PHFHEADER_BITSTREAM]};

    auto start = ((uint8_t*)memcpy_start + memcpy_adjust_to_start);

    auto d2d_memcpy_merge = [&](memcpy_helper& var) {
        FZ_CUDA_CHECK(cudaMemcpyAsync(start + var.dst, var.ptr, var.nbyte,
                                      cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
    };

    FZ_CUDA_CHECK(cudaMemcpyAsync(start, &header, sizeof(header),
                                  cudaMemcpyHostToDevice, (cudaStream_t)stream));

    d2d_memcpy_merge(_revbk);
    d2d_memcpy_merge(_par_nbit);
    d2d_memcpy_merge(_par_entry);
    d2d_memcpy_merge(_bitstream);
}

template <typename E>
void Buf<E>::clear_buffer()
{
    cudaMemset(d_scratch4,   0, len);
    cudaMemset(d_bk4,        0, bklen);
    cudaMemset(d_revbk4,     0, revbk4_bytes);
    cudaMemset(d_bitstream4, 0, bitstream_max_len);
    cudaMemset(d_par_nbit,   0, pardeg);
    cudaMemset(d_par_ncell,  0, pardeg);
    cudaMemset(d_par_entry,  0, pardeg);
}

}  // namespace phf

template struct phf::Buf<uint8_t>;
template struct phf::Buf<uint16_t>;
template struct phf::Buf<uint32_t>;
