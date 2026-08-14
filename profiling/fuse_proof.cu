// Go/no-go proof: fully-fused cuSZp2 compress vs the staged FZGM pipeline.
//
// The staged fast pipeline (Quantizer -> Lorenzo(32) -> AdaptiveBitpack) reads
// the input once but then materialises int32 codes to DRAM and re-reads them
// across the Lorenzo, rate, and pack kernels. This prototype fuses the entire
// block-local chain: one warp owns one 32-element block and computes quant +
// Lorenzo delta + adaptive-bitpack entirely in registers, so the int32 codes are
// never written to or read from DRAM. It keeps the well-optimised CUB offset
// scan (recomputing quant+Lorenzo in the pack kernel rather than materialising
// deltas), which sidesteps the in-kernel look-back that made the earlier
// rate+pack-only fusion barrier-bound (see CN-AB-FUSE).
//
// This measures the *ceiling* of fusion for the go/no-go decision: does removing
// the code round-trips move throughput meaningfully toward native cuSZp2, or
// does the pipeline plateau? Correctness is a full round-trip (decode + inverse
// Lorenzo + inverse quant) checked against the error bound.
//
// Usage: fzgmod-profile-fuse [total_MB] [eb] [runs] [file]

#include "fzgpumodules.h"
#include "pipeline/perf.h"
#include "coders/adaptive_bitpack/adaptive_bitpack_kernels.h"
#include "predictors/lorenzo/lorenzo_stage.h"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace fz;
namespace ab = fz::adaptive_bitpack;
using Clock = std::chrono::high_resolution_clock;

static const char* DATA_PATH = FZ_DATA_DIR "/CLDHGH.f32";
static constexpr size_t SRC_ELEMS = 3600 * 1800;

// ── Device helpers (self-contained copies of the AB primitives) ──────────────
__device__ __forceinline__ uint32_t absU_i32(int v) {
    return static_cast<uint32_t>(v < 0 ? -v : v);
}
__device__ __forceinline__ int bitWidth32(uint32_t x) { return x ? (32 - __clz(x)) : 0; }

// Quant (linear ABS) + 1-D Lorenzo delta (reset per 32-block) for lane `l` of
// block `b`. Produces the signed delta this warp's element contributes, and
// whether it is in range. Byte-identical to Quantizer(linear)->Lorenzo(32).
__device__ __forceinline__ int computeDelta(
    const float* __restrict__ in, size_t n, float inv2eb,
    size_t b, uint32_t l, bool& active)
{
    const size_t gidx = b * 32u + l;
    active = gidx < n;
    const int q = active ? __float2int_rn(in[gidx] * inv2eb) : 0;
    const int qprev = __shfl_up_sync(0xffffffffu, q, 1);
    return (l == 0) ? q : (q - qprev);        // block reset at lane 0
}

// ── Fused kernel A: quant + Lorenzo + AB outlier rate/cost ───────────────────
__global__ void fused_rate_kernel(
    const float* __restrict__ in, size_t n, float inv2eb, uint32_t word_bytes,
    size_t num_blocks, uint8_t* __restrict__ meta, uint32_t* __restrict__ cost)
{
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const size_t b = static_cast<size_t>(blockIdx.x) * (blockDim.x >> 5) + warp;
    if (b >= num_blocks) return;

    bool active;
    const int delta = computeDelta(in, n, inv2eb, b, lane, active);
    const uint32_t av = active ? absU_i32(delta) : 0u;

    uint32_t acc_all = av;
    uint32_t acc_rest = (lane > 0) ? av : 0u;   // element 0 excluded from "rest"
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc_all  |= __shfl_xor_sync(0xffffffffu, acc_all, off);
        acc_rest |= __shfl_xor_sync(0xffffffffu, acc_rest, off);
    }
    if (lane == 0) {
        const uint32_t mag0 = av;                 // element 0's magnitude
        const int fr_all  = bitWidth32(acc_all);
        const int fr_rest = bitWidth32(acc_rest);
        const uint32_t ob_bytes = static_cast<uint32_t>((bitWidth32(mag0) + 7) / 8);
        const uint32_t cost_plain = (fr_all > 0) ? word_bytes * (fr_all + 1u) : 0u;
        const uint32_t cost_out   = ob_bytes
                                  + ((fr_rest > 0) ? word_bytes * (fr_rest + 1u) : word_bytes);
        if (cost_plain <= cost_out) { meta[2*b]=fr_all;  meta[2*b+1]=0; cost[b]=cost_plain; }
        else { meta[2*b]=fr_rest; meta[2*b+1]=static_cast<uint8_t>(1u | ((ob_bytes-1u)<<1)); cost[b]=cost_out; }
    }
}

// ── Fused kernel B: quant + Lorenzo + AB outlier pack (gather) ────────────────
// Gather pack (O(rate)); produces the identical archive to the transpose path.
__global__ void fused_pack_kernel(
    const float* __restrict__ in, size_t n, float inv2eb, uint32_t word_bytes,
    size_t num_blocks, const uint8_t* __restrict__ meta,
    const uint32_t* __restrict__ offset, uint8_t* __restrict__ payload)
{
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const size_t b = static_cast<size_t>(blockIdx.x) * (blockDim.x >> 5) + warp;
    if (b >= num_blocks) return;

    const int     r   = meta[2*b];
    const uint8_t sel = meta[2*b+1];
    const bool is_out = (sel & 1u) != 0;

    bool active;
    const int delta = computeDelta(in, n, inv2eb, b, lane, active);
    const uint32_t av = active ? absU_i32(delta) : 0u;
    const bool neg = active && delta < 0;
    uint8_t* base = payload + offset[b];

    if (!is_out) {
        if (r == 0) return;
        const uint32_t sm = __ballot_sync(0xffffffffu, neg);
        if (lane < 4) base[lane] = static_cast<uint8_t>((sm >> (8u*lane)) & 0xFFu);
        for (int p = 0; p < r; ++p) {
            const uint32_t pm = __ballot_sync(0xffffffffu, active && ((av >> p) & 1u));
            if (lane < 4) base[word_bytes*(1u+p)+lane] = static_cast<uint8_t>((pm >> (8u*lane)) & 0xFFu);
        }
        return;
    }
    // Outlier block: [ob_bytes elem0 mag][sign][r planes for elems 1..].
    const uint32_t ob_bytes = ((sel >> 1) & 3u) + 1u;
    if (lane == 0) for (uint32_t k = 0; k < ob_bytes; ++k)
        base[k] = static_cast<uint8_t>((av >> (8u*k)) & 0xffu);
    uint8_t* sign   = base + ob_bytes;
    uint8_t* planes = base + ob_bytes + word_bytes;
    const bool plane_active = active && lane > 0;
    const uint32_t sm = __ballot_sync(0xffffffffu, neg);
    if (lane < 4) sign[lane] = static_cast<uint8_t>((sm >> (8u*lane)) & 0xFFu);
    for (int p = 0; p < r; ++p) {
        const uint32_t pm = __ballot_sync(0xffffffffu, plane_active && ((av >> p) & 1u));
        if (lane < 4) planes[word_bytes*p+lane] = static_cast<uint8_t>((pm >> (8u*lane)) & 0xFFu);
    }
}

__global__ void inv_quant_kernel(const int* codes, size_t n, float ebx2, float* out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = codes[i] * ebx2;
}

// ── Host ─────────────────────────────────────────────────────────────────────
static double ms(std::chrono::time_point<Clock> a, std::chrono::time_point<Clock> b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static double gbs(size_t bytes, double m) { return double(bytes) / (m * 1e-3) / 1e9; }

static void build_cuszp2(Pipeline& p, float eb, size_t n) {
    p.setDims(n, 1, 1);
    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(eb); q->setErrorBoundMode(ErrorBoundMode::ABS); q->setLinearMode(true);
    auto* l = p.addStage<LorenzoStage<int32_t>>(); l->setBlockSize(32);
    p.connect(l, q, "codes");
    auto* a = p.addStage<AdaptiveBitpackStage<int32_t>>();
    a->setBlockSize(32); a->setOutlierSelection(true);
    p.connect(a, l);
}

int main(int argc, char** argv) {
    size_t total_mb = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 256;
    float  eb       = (argc > 2) ? std::strtof(argv[2], nullptr)      : 1e-3f;
    int    runs     = (argc > 3) ? std::atoi(argv[3])                 : 10;
    const char* file = (argc > 4) ? argv[4] : nullptr;

    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s   L2=%.0f MB\n", prop.name, prop.l2CacheSize / 1048576.0);

    // Load field (real file or replicated CLDHGH).
    size_t total_elems = (total_mb * 1048576ull) / sizeof(float);
    std::vector<float> src;
    if (file) {
        FILE* f = std::fopen(file, "rb");
        if (!f) { std::fprintf(stderr, "open %s\n", file); return 1; }
        std::fseek(f, 0, SEEK_END); size_t fe = std::ftell(f)/sizeof(float); std::fseek(f,0,SEEK_SET);
        total_elems = std::min(total_elems, fe); src.resize(total_elems);
        if (std::fread(src.data(), sizeof(float), total_elems, f) != total_elems) return 1;
        std::fclose(f);
    } else {
        src.resize(SRC_ELEMS);
        FILE* f = std::fopen(DATA_PATH, "rb");
        if (!f || std::fread(src.data(), sizeof(float), SRC_ELEMS, f) != SRC_ELEMS) return 1;
        std::fclose(f);
    }
    const size_t n = total_elems;
    const size_t in_bytes = n * sizeof(float);
    float* d_in = nullptr; cudaMalloc(&d_in, in_bytes);
    for (size_t o = 0; o < n; o += src.size())
        cudaMemcpy(d_in + o, src.data(), std::min(src.size(), n - o) * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    printf("Field: %.1f MB (%zu elems)  eb=%.0e  runs=%d\n\n", in_bytes/1048576.0, n, eb, runs);

    // ── Staged baseline (FZGM Pipeline) ─────────────────────────────────────
    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE, 2.0f);
    build_cuszp2(p, eb, n); p.enableProfiling(true); p.finalize();
    void* d_comp = nullptr; size_t comp_sz = 0;
    double staged_best = 1e30, staged_dev = 0;
    for (int r = 0; r < runs + 1; ++r) {
        cudaDeviceSynchronize(); auto t0 = Clock::now();
        p.compress(d_in, in_bytes, &d_comp, &comp_sz, 0);
        cudaDeviceSynchronize(); auto t1 = Clock::now();
        if (r > 0 && ms(t0,t1) < staged_best) { staged_best = ms(t0,t1); staged_dev = p.getLastPerfResult().dag_elapsed_ms; }
    }
    printf("staged   wall %7.3f ms  %7.2f GB/s | dev %7.3f ms  %7.2f GB/s   comp=%.2f MB\n",
           staged_best, gbs(in_bytes, staged_best), staged_dev, gbs(in_bytes, staged_dev), comp_sz/1048576.0);

    // ── Fused compress ──────────────────────────────────────────────────────
    ab::Config cfg = ab::configure(n, 32, /*outlier=*/true);
    const float inv2eb = 1.0f / (2.0f * eb);
    const size_t meta_region = cfg.meta_bytes * cfg.num_blocks;
    uint8_t* d_archive = nullptr; cudaMalloc(&d_archive, ab::maxArchiveBytes(cfg, 8u*sizeof(int32_t)));
    uint32_t* d_cost = nullptr; uint32_t* d_offset = nullptr;
    cudaMalloc(&d_cost, cfg.num_blocks*4); cudaMalloc(&d_offset, cfg.num_blocks*4);
    size_t cub_tmp_bytes = 0; cub::DeviceScan::ExclusiveSum(nullptr, cub_tmp_bytes, d_cost, d_offset, cfg.num_blocks);
    void* d_cub = nullptr; cudaMalloc(&d_cub, cub_tmp_bytes);

    const int WPB = 8, THREADS = WPB*32;
    const int grid = static_cast<int>((cfg.num_blocks + WPB - 1) / WPB);
    auto fused_once = [&](cudaStream_t s) {
        uint8_t* d_meta = d_archive; uint8_t* d_payload = d_archive + meta_region;
        fused_rate_kernel<<<grid, THREADS, 0, s>>>(d_in, n, inv2eb, cfg.word_bytes, cfg.num_blocks, d_meta, d_cost);
        cub::DeviceScan::ExclusiveSum(d_cub, cub_tmp_bytes, d_cost, d_offset, cfg.num_blocks, s);
        fused_pack_kernel<<<grid, THREADS, 0, s>>>(d_in, n, inv2eb, cfg.word_bytes, cfg.num_blocks, d_meta, d_offset, d_payload);
    };
    double fused_best = 1e30;
    for (int r = 0; r < runs + 1; ++r) {
        cudaDeviceSynchronize(); auto t0 = Clock::now();
        fused_once(0);
        cudaDeviceSynchronize(); auto t1 = Clock::now();
        if (r > 0) fused_best = std::min(fused_best, ms(t0,t1));
    }
    // fused compressed size = meta + total payload (offset[last] + cost[last]).
    uint32_t h_off = 0, h_cost = 0;
    cudaMemcpy(&h_off, d_offset + cfg.num_blocks-1, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_cost, d_cost + cfg.num_blocks-1, 4, cudaMemcpyDeviceToHost);
    size_t fused_sz = meta_region + h_off + h_cost;
    printf("fused    wall %7.3f ms  %7.2f GB/s | (single-kernel-launch chain)         comp=%.2f MB\n",
           fused_best, gbs(in_bytes, fused_best), fused_sz/1048576.0);

    // ── Correctness: round-trip the fused archive ───────────────────────────
    int* d_deltas = nullptr; int* d_codes = nullptr; float* d_recon = nullptr;
    cudaMalloc(&d_deltas, n*4); cudaMalloc(&d_codes, n*4); cudaMalloc(&d_recon, in_bytes);
    ab::launchDecodeUnpackOutlier<int32_t>(d_archive, d_offset, d_archive + meta_region, cfg, d_deltas, 0);
    fz::launchLorenzoPrefixSumKernel1D<int32_t>(d_deltas, d_codes, n, 0, 32u);
    inv_quant_kernel<<<(n+255)/256, 256>>>(d_codes, n, 2.0f*eb, d_recon);
    std::vector<float> h_recon(n); cudaMemcpy(h_recon.data(), d_recon, in_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    double maxerr = 0; for (size_t i = 0; i < n; ++i) maxerr = std::max(maxerr, (double)std::abs(h_recon[i] - src[i % src.size()]));

    printf("\nfused speedup vs staged: wall %.2fx   dev %.2fx\n",
           staged_best/fused_best, staged_dev/fused_best);
    printf("round-trip max error = %.3e  (eb=%.3e)  %s\n", maxerr, (double)eb,
           maxerr <= eb*1.001 ? "[OK within bound]" : "[FAIL over bound]");
    printf("compressed size match: fused=%.3f MB  staged(incl header)=%.3f MB  %s\n",
           fused_sz/1048576.0, comp_sz/1048576.0,
           (comp_sz >= fused_sz && comp_sz - fused_sz < 4096) ? "[match +header]" : "[MISMATCH]");

    cudaFree(d_in); cudaFree(d_archive); cudaFree(d_cost); cudaFree(d_offset); cudaFree(d_cub);
    cudaFree(d_deltas); cudaFree(d_codes); cudaFree(d_recon);
    return 0;
}
