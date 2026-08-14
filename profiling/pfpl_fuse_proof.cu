// Go/no-go proof: chunk-cooperative fused PFPL compress vs the staged pipeline.
//
// PFPL = Quantizer(NOA,zigzag) -> Difference(1-D chunk, negabinary) ->
//        Bitshuffle(ew=4) -> RZE(w=1). All three post-quant stages chunk at
//        16384 bytes = 4096 int32 elements = one CTA. This fuses them into ONE
//        CTA-per-chunk kernel that keeps the quant codes / diff residuals /
//        bitshuffled bytes in shared memory (no DRAM round-trips), then keeps the
//        cross-chunk scan+pack tail (RZE offsets depend on all chunks).
//
// KEY OCCUPANCY POINT: the fused kernel reuses RZE's own two 16 KB smem buffers
// for every intermediate, so its footprint (2x16 KB + 4 KB temp = 36 KB) equals
// rzeEncodeKernel alone — fusion adds ZERO smem, so no occupancy collapse.
//
// The stage device functions are reused verbatim (Zigzag/Negabinary encode,
// butterfly32 copied from bitshuffle, lc_detail::d_RZE) with the quantizer's
// exact ebx2_r (matched via a forced value_base), so the fused archive is
// byte-identical to a manually-run staged forward. Validated + timed here.
//
// Restriction (prototype): outlier-free fast path (verified: NOA/CLDHGH codes
// << radius, outlier_count==0), and n truncated to a multiple of 4096 elements
// so every chunk is full (no partial-tail handling). See CN-PFPL-FUSE.

#include "fzgpumodules.h"
#include "quantizers/quantizer/quantizer.h"
#include "predictors/diff/diff.h"
#include "shufflers/bitshuffle/bitshuffle_stage.h"
#include "coders/rze/rze_stage.h"
#include "coders/lc_common/lc_chunk_components.cuh"
#include "transforms/zigzag/zigzag.h"
#include "transforms/negabinary/negabinary.h"
#include "mem/mempool.h"

#include "backend/cub.h"
#include <cub/device/device_scan.cuh>
#include <thrust/iterator/transform_iterator.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace fz;
using byte  = uint8_t;
using Clock = std::chrono::high_resolution_clock;

static const char*      DATA_PATH = FZ_DATA_DIR "/CLDHGH.f32";
static constexpr size_t DIM_X = 3600, DIM_Y = 1800;
static constexpr int    CS    = 16384;           // chunk bytes = RZE/bitshuffle/diff chunk
static constexpr int    NELEM = CS / 4;          // 4096 int32 codes per chunk
static constexpr int    NPP   = NELEM / 32;      // 128 bitshuffle words per plane
static constexpr int    RZE_TEMP = 4096, TPB = 512;

static double ms(std::chrono::time_point<Clock> a, std::chrono::time_point<Clock> b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static double gbs(size_t bytes, double milli) { return bytes / (milli * 1e-3) / 1e9; }

// 5-stage register butterfly (copy of bitshuffle_stage.cu butterfly32).
__device__ __forceinline__ unsigned butterfly32(unsigned a, int sublane) {
    unsigned q = __shfl_xor_sync(0xffffffffu, a, 16, 32);
    a = ((sublane&16)==0) ? __byte_perm(a,q,(3u<<12)|(2u<<8)|(7u<<4)|6u)
                          : __byte_perm(a,q,(5u<<12)|(4u<<8)|(1u<<4)|0u);
    q = __shfl_xor_sync(0xffffffffu, a, 8, 32);
    a = ((sublane&8)==0) ? __byte_perm(a,q,(3u<<12)|(7u<<8)|(1u<<4)|5u)
                         : __byte_perm(a,q,(6u<<12)|(2u<<8)|(4u<<4)|0u);
    q = __shfl_xor_sync(0xffffffffu, a, 4, 32); unsigned m=0x0F0F0F0Fu;
    a = ((sublane&4)==0) ? ((a&~m)|((q>>4)&m)) : (((q<<4)&~m)|(a&m));
    q = __shfl_xor_sync(0xffffffffu, a, 2, 32); m=0x33333333u;
    a = ((sublane&2)==0) ? ((a&~m)|((q>>2)&m)) : (((q<<2)&~m)|(a&m));
    q = __shfl_xor_sync(0xffffffffu, a, 1, 32); m=0x55555555u;
    a = ((sublane&1)==0) ? ((a&~m)|((q>>1)&m)) : (((q<<1)&~m)|(a&m));
    return a;
}

// One CTA per 4096-element chunk: quant -> diff+negabinary -> bitshuffle -> RZE.
__global__ void __launch_bounds__(TPB)
pfpl_fused_encode(const float* __restrict__ in, size_t n_elems, float ebx2_r,
                  byte* __restrict__ d_scratch, uint32_t* __restrict__ d_sizes)
{
    __shared__ __align__(16) uint32_t sA[NELEM];   // 16 KB: codes -> (reused) bitshuffled
    __shared__ __align__(16) byte     sB[CS];      // 16 KB: negabinary -> RZE s_out
    __shared__ __align__(16) byte     sTemp[RZE_TEMP];

    const uint32_t cid  = blockIdx.x;
    const size_t   base = (size_t)cid * NELEM;
    const int      tid  = threadIdx.x;
    const int      lane = tid & 31;

    // (1) Quantize (NOA/ABS + zigzag) -> sA codes.
    for (int i = tid; i < NELEM; i += TPB) {
        const size_t g = base + i;
        const float  x = (g < n_elems) ? in[g] : 0.0f;
        const int    q = __float2int_rn(x * ebx2_r);
        sA[i] = Zigzag<int32_t>::encode(q);
    }
    __syncthreads();

    // (2) Chunk-local difference (boundary = element 0) + negabinary -> sB (uint32).
    uint32_t* sBu = reinterpret_cast<uint32_t*>(sB);
    for (int i = tid; i < NELEM; i += TPB) {
        const int ci = (int)sA[i];
        const int d  = (i == 0) ? ci : (ci - (int)sA[i-1]);
        sBu[i] = Negabinary<int32_t>::encode(d);
    }
    __syncthreads();

    // (3) Bitshuffle sBu (4096 codes) -> sA in bit-plane layout (smem scatter).
    for (int i = tid; i < NELEM; i += TPB)
        sA[i/32 + lane*NPP] = butterfly32(sBu[i], lane);
    __syncthreads();

    // (4) RZE-encode the bitshuffled bytes in place (reuses d_RZE verbatim).
    byte* rze_in = reinterpret_cast<byte*>(sA);   // 16 KB bitshuffled input
    int   csize  = CS;
    bool  good   = lc_detail::d_RZE<byte, CS>(csize, rze_in, sB, sTemp);
    __syncthreads();

    byte* out = d_scratch + (size_t)cid * CS;
    if (good && csize < CS) {
        for (int i = tid; i < csize; i += TPB) out[i] = sB[i];
        if (tid == 0) d_sizes[cid] = (uint32_t)csize;
    } else {
        for (int i = tid; i < CS; i += TPB) out[i] = rze_in[i];
        if (tid == 0) d_sizes[cid] = (1u << 31) | (uint32_t)CS;
    }
}

// Pack compressed chunks from uniform scratch to the packed archive payload.
__global__ void pfpl_pack(const byte* __restrict__ scratch, byte* __restrict__ outp,
                          const uint32_t* __restrict__ off, const uint32_t* __restrict__ sz,
                          uint32_t header) {
    const uint32_t cid = blockIdx.x;
    const uint32_t n   = sz[cid] & 0x7FFFFFFFu;
    const byte* s = scratch + (size_t)cid * CS;
    byte* d = outp + header + off[cid];
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i];
}

struct StripFlag { __host__ __device__ uint32_t operator()(uint32_t x) const { return x & 0x7FFFFFFFu; } };

int main(int argc, char** argv) {
    const int   runs = argc > 1 ? atoi(argv[1]) : 15;
    const float eb   = argc > 2 ? (float)atof(argv[2]) : 1e-4f;

    // ── Load CLDHGH, truncate to a whole number of 4096-element chunks ──────────
    size_t n_full = DIM_X * DIM_Y;
    std::vector<float> h(n_full);
    { FILE* fp = fopen(DATA_PATH, "rb");
      if (!fp) { printf("cannot open %s\n", DATA_PATH); return 1; }
      if (fread(h.data(), 4, n_full, fp) != n_full) { printf("short read\n"); return 1; }
      fclose(fp); }
    const size_t n  = (n_full / NELEM) * NELEM;    // drop the partial tail chunk
    const size_t nc = n / NELEM;
    h.resize(n);
    const size_t in_bytes = n * sizeof(float);

    // NOA scale: force value_base so staged and fused share the exact ebx2_r.
    float vmin = h[0], vmax = h[0];
    for (float v : h) { vmin = std::min(vmin, v); vmax = std::max(vmax, v); }
    const float value_base = vmax - vmin;
    const float abs_eb = eb * value_base;
    const float ebx2_r = 1.0f / (2.0f * abs_eb);

    printf("PFPL fused proof — CLDHGH  n=%zu (%zu chunks, dropped %zu tail elems)  eb=%.1e NOA\n",
           n, nc, n_full - n, (double)eb);
    printf("value_base=%.6g  abs_eb=%.6g  ebx2_r=%.6g\n\n", value_base, abs_eb, ebx2_r);

    float* d_in = nullptr; cudaMalloc(&d_in, in_bytes);
    cudaMemcpy(d_in, h.data(), in_bytes, cudaMemcpyHostToDevice);

    MemoryPool pool(MemoryPoolConfig{});
    cudaStream_t stream = 0;

    // ── Staged reference: run the 4 forward stages manually → raw RZE archive ───
    auto* d_codes = (uint32_t*)nullptr; cudaMalloc(&d_codes, n * 4);
    auto* d_oval  = (float*)nullptr;    cudaMalloc(&d_oval,  (size_t)(n * 0.2) * 4 + 16);
    auto* d_oidx  = (uint32_t*)nullptr; cudaMalloc(&d_oidx,  (size_t)(n * 0.2) * 4 + 16);
    auto* d_nb    = (uint32_t*)nullptr; cudaMalloc(&d_nb, n * 4);
    auto* d_bs    = (uint32_t*)nullptr; cudaMalloc(&d_bs, n * 4);
    auto* d_rze   = (byte*)nullptr;     cudaMalloc(&d_rze, in_bytes + 4096);

    QuantizerStage<float, uint32_t> q;
    q.setErrorBound(eb); q.setErrorBoundMode(ErrorBoundMode::NOA);
    q.setValueBase(value_base); q.setQuantRadius(32768);
    q.setOutlierCapacity(0.2f); q.setZigzagCodes(true);
    DifferenceStage<int32_t, uint32_t> df; df.setChunkSize(CS);
    BitshuffleStage bs; bs.setElementWidth(4); bs.setBlockSize(CS);
    RZEStage rz; rz.setWordSize(1); rz.setChunkSize(CS);

    { std::vector<void*> in_{d_in}, out_{d_codes, d_oval, d_oidx}; std::vector<size_t> sz_{in_bytes};
      q.execute(stream, &pool, in_, out_, sz_); }
    { std::vector<void*> in_{d_codes}, out_{d_nb}; std::vector<size_t> sz_{n*4};
      df.execute(stream, &pool, in_, out_, sz_); }
    { std::vector<void*> in_{d_nb}, out_{d_bs}; std::vector<size_t> sz_{n*4};
      bs.execute(stream, &pool, in_, out_, sz_); }
    { std::vector<void*> in_{d_bs}, out_{d_rze}; std::vector<size_t> sz_{n*4};
      rz.execute(stream, &pool, in_, out_, sz_); rz.postStreamSync(stream); }
    cudaDeviceSynchronize();
    const size_t staged_bytes = rz.getActualOutputSize(0);
    printf("staged (manual 4-stage): RZE archive=%.3f MB (ratio %.2fx)\n",
           staged_bytes / 1048576.0, (double)in_bytes / staged_bytes);
    // (Any quantizer outliers would leave 0-substituted codes in the staged path
    //  that the outlier-free fused path cannot match → caught by the byte compare.)
    std::vector<byte> A_staged(staged_bytes);
    cudaMemcpy(A_staged.data(), d_rze, staged_bytes, cudaMemcpyDeviceToHost);

    // ── Fused path: encode kernel + exclusive-scan offsets + pack ───────────────
    byte*     d_scratch = nullptr; cudaMalloc(&d_scratch, nc * CS);
    uint32_t* d_sz      = nullptr; cudaMalloc(&d_sz,  nc * 4);
    uint32_t* d_off     = nullptr; cudaMalloc(&d_off, nc * 4);
    byte*     d_arch    = nullptr; cudaMalloc(&d_arch, in_bytes + 4096);
    const uint32_t header = 8 + 4 * (uint32_t)nc;

    auto fused_once = [&]() -> size_t {
        pfpl_fused_encode<<<(int)nc, TPB, 0, stream>>>(d_in, n, ebx2_r, d_scratch, d_sz);
        // header: [orig_total][num_chunks][per-chunk sizes]
        uint32_t hdr[2] = { (uint32_t)in_bytes, (uint32_t)nc };
        cudaMemcpyAsync(d_arch, hdr, 8, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_arch + 8, d_sz, nc * 4, cudaMemcpyDeviceToDevice, stream);
        auto strip = thrust::make_transform_iterator((const uint32_t*)d_sz, StripFlag{});
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, strip, d_off, (int)nc, stream);
        void* d_tmp = nullptr; cudaMalloc(&d_tmp, tmp_bytes);
        cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, strip, d_off, (int)nc, stream);
        pfpl_pack<<<(int)nc, 256, 0, stream>>>(d_scratch, d_arch, d_off, d_sz, header);
        uint32_t last_off = 0, last_sz = 0;
        cudaMemcpyAsync(&last_off, d_off + nc - 1, 4, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&last_sz,  d_sz  + nc - 1, 4, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_tmp);
        const size_t total = header + last_off + (last_sz & 0x7FFFFFFFu);
        const size_t padded = (total + 3) & ~size_t(3);   // match RZE's 4-byte round + zero pad
        if (padded > total)
            cudaMemsetAsync(d_arch + total, 0, padded - total, stream);
        cudaStreamSynchronize(stream);
        return padded;
    };

    const size_t fused_bytes = fused_once();
    cudaDeviceSynchronize();
    std::vector<byte> A_fused(fused_bytes);
    cudaMemcpy(A_fused.data(), d_arch, fused_bytes, cudaMemcpyDeviceToHost);

    // ── Correctness: byte-identical archive ─────────────────────────────────────
    bool size_ok = (fused_bytes == staged_bytes);
    bool byte_ok = size_ok && (memcmp(A_fused.data(), A_staged.data(), fused_bytes) == 0);
    size_t first_diff = fused_bytes;
    if (size_ok && !byte_ok)
        for (size_t i = 0; i < fused_bytes; ++i) if (A_fused[i] != A_staged[i]) { first_diff = i; break; }
    printf("\nfused archive=%.3f MB (ratio %.2fx)\n", fused_bytes / 1048576.0, (double)in_bytes / fused_bytes);
    printf("byte-identical to staged: %s%s\n", byte_ok ? "YES [OK]" : "NO [FAIL]",
           size_ok ? "" : "  (size mismatch)");
    if (size_ok && !byte_ok) printf("  first differing byte at %zu\n", first_diff);

    // ── Timing: fused encode+scan+pack vs staged compress DAG ───────────────────
    auto timeit = [&](auto fn) { double best = 1e30;
        for (int r = 0; r < runs; ++r) { auto t0 = Clock::now(); fn(); cudaDeviceSynchronize();
            best = std::min(best, ms(t0, Clock::now())); } return best; };
    double fused_ms = timeit([&]{ fused_once(); });

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE, 4.0f);
    p.setDims(n, 1, 1);
    { auto* pq = p.addStage<QuantizerStage<float,uint32_t>>();
      pq->setErrorBound(eb); pq->setErrorBoundMode(ErrorBoundMode::NOA);
      pq->setValueBase(value_base); pq->setQuantRadius(32768);
      pq->setOutlierCapacity(0.2f); pq->setZigzagCodes(true);
      auto* pd = p.addStage<DifferenceStage<int32_t,uint32_t>>(); pd->setChunkSize(CS);
      p.connect(pd, pq, "codes");
      auto* pb = p.addStage<BitshuffleStage>(); pb->setElementWidth(4); pb->setBlockSize(CS);
      p.connect(pb, pd);
      auto* pr = p.addStage<RZEStage>(); pr->setWordSize(1); pr->setChunkSize(CS);
      p.connect(pr, pb); }
    p.enableProfiling(true);
    p.finalize();
    void* d_comp = nullptr; size_t comp_sz = 0;
    double staged_dev = 1e30, staged_ms = timeit([&]{
        p.compress(d_in, in_bytes, &d_comp, &comp_sz, 0); });
    for (int r = 0; r < runs; ++r) { p.compress(d_in, in_bytes, &d_comp, &comp_sz, 0);
        cudaDeviceSynchronize(); staged_dev = std::min(staged_dev, (double)p.getLastPerfResult().dag_elapsed_ms); }

    printf("\n─ throughput (compress) ─────────────────────────────\n");
    printf("staged  wall %.3f ms (%.1f GB/s)   dag %.3f ms (%.1f GB/s)\n",
           staged_ms, gbs(in_bytes, staged_ms), staged_dev, gbs(in_bytes, staged_dev));
    printf("fused   wall %.3f ms (%.1f GB/s)\n", fused_ms, gbs(in_bytes, fused_ms));
    printf("speedup vs staged wall %.2fx   vs staged dag %.2fx\n",
           staged_ms / fused_ms, staged_dev / fused_ms);
    printf("\nfused kernel smem/block = %d B (%.0f KB)  [RZE-encode alone = same]\n",
           2*CS + RZE_TEMP, (2*CS + RZE_TEMP)/1024.0);

    cudaFree(d_in); cudaFree(d_codes); cudaFree(d_oval); cudaFree(d_oidx);
    cudaFree(d_nb); cudaFree(d_bs); cudaFree(d_rze);
    cudaFree(d_scratch); cudaFree(d_sz); cudaFree(d_off); cudaFree(d_arch);
    return byte_ok ? 0 : 2;
}
