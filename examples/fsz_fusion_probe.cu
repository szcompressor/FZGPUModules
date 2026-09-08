/**
 * examples/fsz_fusion_probe.cu
 *
 * EXPERIMENTAL probe, not a shipped stage: measures whether folding the
 * upstream linear Quantizer's inline quantization into AdaptiveLorenzo's
 * forward kernel ("M1" partial fusion, see memory/generic_fusion_plan.md's
 * ARCHITECTURAL TENSION section, approach C) is worth pursuing at the field
 * sizes where FSZ actually runs (tens to hundreds of MB), as opposed to the
 * 25 MB CLDHGH field the earlier checkpoint-4 gate used exclusively.
 *
 * What it does:
 *   1. Loads a raw float32 field.
 *   2. Runs the REAL staged Quantizer(linear,NOA) -> AdaptiveLorenzo path by
 *      calling QuantizerStage::execute() then AdaptiveLorenzoStage::execute()
 *      directly (unmodified library code, bypassing Pipeline/DAG so the raw
 *      per-stage device buffers — residuals, packed modes, compacted means —
 *      are directly in hand for comparison, since Pipeline only exposes named
 *      output buffers through a private API intended for its own file I/O).
 *   3. Runs a NEW fused kernel that reads the raw floats directly and
 *      quantizes inline (deleting the Quantizer kernel + its codes[] DRAM
 *      round-trip), reusing the exact same difference-chain / 4-variant
 *      selection arithmetic as the staged kernel, then reuses the REAL
 *      launchAdaptiveLorenzoCompact<T>() for mode/mean compaction so that
 *      step stays byte-for-byte shared between both paths.
 *   4. Compares residuals / packed modes / compacted means BYTE-FOR-BYTE
 *      between the two paths (this is a stronger check than round-trip: the
 *      fused kernel's arithmetic is a line-for-line copy of the staged
 *      kernel's, so any divergence is a bug, not a design tradeoff).
 *   5. Times the fused kernel + compaction step (cudaEvent, warmup + N reps,
 *      median) and reports achieved GB/s so it can be compared directly
 *      against the staged Quantizer+AdaptiveLorenzo stage times reported by
 *      `fzgmod-cli --profile`.
 *
 * This is intentionally NOT wired into the Stage/DAG/fusion-registry
 * machinery — it exists to answer one question (does removing the Quantizer
 * pass matter at scale) before investing in that wiring.
 *
 * Usage:
 *   ./fsz_fusion_probe <field.f32> [--eb <rel>] [--reps <n>]
 *
 * Build:
 *   cmake -B build_fusion_probe -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release \
 *         -DCMAKE_CUDA_ARCHITECTURES=90
 *   cmake --build build_fusion_probe --target fsz_fusion_probe -j
 */

#include "fzgpumodules.h"
#include "quantizers/quantizer/quantizer.h"
#include "fused/adaptive_lorenzo/adaptive_lorenzo_stage.h"
#include "mem/mempool.h"
#include "coders/adaptive_bitpack/adaptive_bitpack_oracle.cuh"
#include "backend/warp.h"
#include "backend/cub.h"
#include "stage/fusion.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(expr) do {                                               \
    cudaError_t _e = (expr);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        std::exit(1);                                                       \
    }                                                                       \
} while (0)

struct DevBuf {
    void*  ptr   = nullptr;
    size_t bytes = 0;
    DevBuf() = default;
    explicit DevBuf(size_t n) : bytes(n) { if (n) CUDA_CHECK(cudaMalloc(&ptr, n)); }
    ~DevBuf() { if (ptr) cudaFree(ptr); }
    DevBuf(DevBuf&& o) noexcept : ptr(o.ptr), bytes(o.bytes) { o.ptr = nullptr; o.bytes = 0; }
    DevBuf& operator=(DevBuf&&) = delete;
    DevBuf(const DevBuf&)       = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    template<typename T> T* as() const { return static_cast<T*>(ptr); }
};

static std::vector<float> read_floats(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    auto sz = f.tellg(); f.seekg(0);
    if (sz % sizeof(float) != 0)
        throw std::runtime_error("File size not a multiple of 4 (expected float32)");
    std::vector<float> v(sz / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), sz);
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused quant+AdaptiveLorenzo forward kernel.
//
// Line-for-line the same arithmetic as adaptive_lorenzo_forward_kernel in
// modules/fused/adaptive_lorenzo/adaptive_lorenzo_stage.cu — the ONLY change
// is the input read: instead of `v = in[gid]` (pre-quantized int32 codes),
// this computes `v = round(raw[gid] * ebx2_r)` inline, matching
// QuantizerStage's quantizer_linear_fwd_kernel (linear_mode, non-high-
// precision path) exactly: scaled = raw * ebx2_r; q = __float2ll_rn(scaled).
// Overflow guarding is intentionally omitted (probe-only, see README note
// below) since none of the fields this is run against are near int32 range.
// ─────────────────────────────────────────────────────────────────────────────
namespace probe {

constexpr uint32_t kMaxBlocksPerTile = 32;
constexpr uint8_t  kModeOrder2       = 0x1;
constexpr uint8_t  kModeCentering    = 0x2;
constexpr uint32_t kNoVariant        = 0xFFFFFFFFu;

template<typename T>
__device__ __forceinline__ typename std::make_unsigned<T>::type absU(T v) {
    using U = typename std::make_unsigned<T>::type;
    U uv = static_cast<U>(v);
    return (v < 0) ? static_cast<U>(~uv + static_cast<U>(1)) : uv;
}

struct CoderStats { uint32_t all; uint32_t rest; uint32_t first; };

__device__ __forceinline__ uint32_t blockCost(
    CoderStats stats, fz::EncodingOracleKind oracle_kind)
{
    switch (oracle_kind) {
        case fz::EncodingOracleKind::AdaptiveFixedRateBitpack:
            return fz::adaptive_bitpack_oracle::quoteAdaptiveFixedRate(
                stats.all, stats.rest, stats.first, /*word_bytes=*/4u).payload_bytes;
        case fz::EncodingOracleKind::PlainFixedRateBitpack:
        default:
            return fz::adaptive_bitpack_oracle::quotePlainFixedRate(
                stats.all, /*word_bytes=*/4u).payload_bytes;
    }
}

__device__ __forceinline__ uint32_t warpOr(uint32_t v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v |= fz::backend::shflXor(v, off, 32);
    return v;
}

template<typename T>
__global__ __launch_bounds__(256, 8) void fused_quant_al_forward_kernel(
    const float* __restrict__ raw,
    float ebx2_r,
    T*       __restrict__ residuals,
    uint8_t* __restrict__ modes,
    T*       __restrict__ means,
    uint32_t* __restrict__ flags,
    size_t n,
    uint32_t tile_size,
    bool enable_order2,
    bool enable_centering,
    fz::EncodingOracleKind oracle_kind)
{
    __shared__ uint32_t  acc1[kMaxBlocksPerTile];
    __shared__ uint32_t  acc2[kMaxBlocksPerTile];
    __shared__ uint32_t  rest1[kMaxBlocksPerTile];
    __shared__ uint32_t  rest2[kMaxBlocksPerTile];
    __shared__ uint32_t  first1[kMaxBlocksPerTile];
    __shared__ uint32_t  first2[kMaxBlocksPerTile];
    __shared__ long long red[kMaxBlocksPerTile];
    __shared__ T         sb_last[kMaxBlocksPerTile];
    __shared__ T         sb_prev[kMaxBlocksPerTile];
    __shared__ T         s_mu;
    __shared__ T         s_q0;
    __shared__ uint8_t   s_mode;

    const size_t   base   = static_cast<size_t>(blockIdx.x) * tile_size;
    const unsigned tid    = threadIdx.x;
    const size_t   gid    = base + tid;
    const bool     live   = (gid < n);
    const unsigned warp   = tid >> 5;
    const unsigned lane   = tid & 31u;
    const unsigned nwarps = tile_size >> 5;

    // ---- THE ONLY CHANGE vs the staged kernel: quantize inline on read ----
    // (32-bit rounding intrinsic, not __float2ll_rn: T is always int32_t/int16_t
    // here and a 64-bit round-trip costs an extra register pair per thread.)
    T v;
    if (live) {
        const float scaled = raw[gid] * ebx2_r;
        v = static_cast<T>(__float2int_rn(scaled));
    } else {
        v = static_cast<T>(0);
    }
    if (tid == 0) s_q0 = v;

    if (enable_centering) {
        long long ssum = live ? static_cast<long long>(v) : 0LL;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ssum += fz::backend::shflDown(ssum, off, 32);
        if (lane == 0u) red[warp] = ssum;
    }

    if (lane == 31u) sb_last[warp] = v;
    if (lane == 30u) sb_prev[warp] = v;

    __syncthreads();

    if (enable_centering) {
        if (warp == 0u) {
            long long t = (lane < nwarps) ? red[lane] : 0LL;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                t += fz::backend::shflDown(t, off, 32);
            if (lane == 0u) {
                const long long count =
                    static_cast<long long>(min(static_cast<size_t>(tile_size), n - base));
                s_mu = static_cast<T>((t >= 0) ? (t + count / 2) / count
                                               : (t - count / 2) / count);
            }
        }
    } else if (tid == 0) {
        s_mu = static_cast<T>(0);
    }

    T vm1 = fz::backend::shflUp(v, 1u, 32);
    if (lane == 0u) vm1 = (warp > 0u) ? sb_last[warp - 1] : static_cast<T>(0);
    const T d1 = static_cast<T>(v - vm1);

    T d1m1 = fz::backend::shflUp(d1, 1u, 32);
    if (lane == 0u)
        d1m1 = (warp > 0u) ? static_cast<T>(sb_last[warp - 1] - sb_prev[warp - 1])
                           : static_cast<T>(0);
    const T d2 = static_cast<T>(d1 - d1m1);

    __syncthreads();

    const T mu = s_mu;
    const T q0 = s_q0;

    const uint32_t m1 = live ? static_cast<uint32_t>(absU<T>(d1)) : 0u;
    const uint32_t m2 = live ? static_cast<uint32_t>(absU<T>(d2)) : 0u;
    const uint32_t o1 = warpOr(m1);
    const uint32_t o2 = warpOr(m2);
    const uint32_t r1rest = warpOr(lane > 0u ? m1 : 0u);
    const uint32_t r2rest = warpOr(lane > 0u ? m2 : 0u);
    if (lane == 0) {
        acc1[warp] = o1;  rest1[warp] = r1rest; first1[warp] = m1;
        acc2[warp] = o2;  rest2[warp] = r2rest; first2[warp] = m2;
    }

    CoderStats c1stats{0u, 0u, 0u}, c2stats{0u, 0u, 0u};
    if (enable_centering && warp == 0u) {
        const T c0 = static_cast<T>(q0 - mu);
        const T cr1 = (tid == 0u) ? c0 : d1;
        T       cr2 = d2;
        if      (tid == 0u) cr2 = c0;
        else if (tid == 1u) cr2 = static_cast<T>(d1 - c0);
        const uint32_t cm1 = live ? static_cast<uint32_t>(absU<T>(cr1)) : 0u;
        const uint32_t cm2 = live ? static_cast<uint32_t>(absU<T>(cr2)) : 0u;
        c1stats.all   = warpOr(cm1);
        c1stats.rest  = warpOr(lane > 0u ? cm1 : 0u);
        c1stats.first = fz::backend::shfl(cm1, 0, 32);
        c2stats.all   = warpOr(cm2);
        c2stats.rest  = warpOr(lane > 0u ? cm2 : 0u);
        c2stats.first = fz::backend::shfl(cm2, 0, 32);
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t c_lz1 = 0, c_lz2 = 0;
        for (unsigned w = 0; w < nwarps; ++w) {
            c_lz1 += blockCost(CoderStats{acc1[w], rest1[w], first1[w]}, oracle_kind);
            c_lz2 += blockCost(CoderStats{acc2[w], rest2[w], first2[w]}, oracle_kind);
        }
        uint32_t costs[4];
        costs[0] = c_lz1;
        costs[1] = enable_order2 ? c_lz2 : kNoVariant;
        costs[2] = kNoVariant;
        costs[3] = kNoVariant;
        if (enable_centering) {
            const uint32_t mean_cost = static_cast<uint32_t>(sizeof(T));
            costs[2] = c_lz1
                - blockCost(CoderStats{acc1[0], rest1[0], first1[0]}, oracle_kind)
                + blockCost(c1stats, oracle_kind) + mean_cost;
            if (enable_order2)
                costs[3] = c_lz2
                    - blockCost(CoderStats{acc2[0], rest2[0], first2[0]}, oracle_kind)
                    + blockCost(c2stats, oracle_kind) + mean_cost;
        }
        uint32_t best = 0;
        for (uint32_t i = 1; i < 4; ++i)
            if (costs[i] < costs[best]) best = i;

        s_mode = static_cast<uint8_t>(((best & 1u) ? kModeOrder2    : 0u)
                                    | ((best & 2u) ? kModeCentering : 0u));
        modes[blockIdx.x] = s_mode;
        means[blockIdx.x] = mu;
        flags[blockIdx.x] = (s_mode & kModeCentering) ? 1u : 0u;
    }
    __syncthreads();

    if (!live) return;
    const uint8_t mode = s_mode;
    const bool    ord2 = (mode & kModeOrder2) != 0;
    const bool    cent = (mode & kModeCentering) != 0;
    const T       c0   = static_cast<T>(q0 - mu);

    T out;
    if (!ord2) {
        out = (cent && tid == 0u) ? c0 : d1;
    } else if (!cent) {
        out = d2;
    } else {
        if      (tid == 0u) out = c0;
        else if (tid == 1u) out = static_cast<T>(d1 - c0);
        else                out = d2;
    }
    residuals[gid] = out;
}

template<typename T>
void launchFusedQuantAL(
    const float* d_raw, float ebx2_r, T* d_residuals, uint8_t* d_modes_dense,
    T* d_means_dense, uint32_t* d_flags, size_t n, uint32_t tile_size,
    bool enable_order2, bool enable_centering, fz::EncodingOracleKind oracle_kind,
    cudaStream_t stream)
{
    const int grid = static_cast<int>((n + tile_size - 1) / tile_size);
    fused_quant_al_forward_kernel<T><<<grid, tile_size, 0, stream>>>(
        d_raw, ebx2_r, d_residuals, d_modes_dense, d_means_dense, d_flags, n,
        tile_size, enable_order2, enable_centering, oracle_kind);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace probe

// The real compaction launcher lives in adaptive_lorenzo_stage.cu at namespace
// fz scope (external linkage, instantiated for int32_t) but isn't declared in
// the public header — declare it here with a matching signature so both the
// staged and fused paths share the exact same compaction step.
namespace fz {
template<typename T>
void launchAdaptiveLorenzoCompact(
    const uint8_t* d_modes_dense, const T* d_means_dense, const uint32_t* d_offsets,
    uint8_t* d_modes_packed, T* d_means_compact, size_t num_tiles,
    cudaStream_t stream);
extern template void launchAdaptiveLorenzoCompact<int32_t>(
    const uint8_t*, const int32_t*, const uint32_t*, uint8_t*, int32_t*, size_t, cudaStream_t);
}

// ─────────────────────────────────────────────────────────────────────────────

struct Args {
    std::string file;
    double      eb_rel = 1e-3;
    int         reps   = 20;
};

static Args parse_args(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <field.f32> [--eb <rel>] [--reps <n>]\n";
        std::exit(1);
    }
    Args a;
    a.file = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--eb" && i + 1 < argc)   a.eb_rel = std::stod(argv[++i]);
        else if (arg == "--reps" && i + 1 < argc) a.reps = std::stoi(argv[++i]);
    }
    return a;
}

int main(int argc, char** argv) {
    using namespace fz;
    Args args = parse_args(argc, argv);

    std::cout << "Loading " << args.file << " ...\n";
    std::vector<float> h_input = read_floats(args.file);
    const size_t n = h_input.size();
    const size_t bytes = n * sizeof(float);
    std::cout << "  " << n << " elements (" << (bytes / 1e6) << " MB)\n";

    cudaStream_t stream = 0;

    DevBuf d_raw(bytes);
    CUDA_CHECK(cudaMemcpy(d_raw.ptr, h_input.data(), bytes, cudaMemcpyHostToDevice));

    // value_base = max - min (NOA), matching QuantizerStage::resolveUniformBound(value_base).
    float vmin = *std::min_element(h_input.begin(), h_input.end());
    float vmax = *std::max_element(h_input.begin(), h_input.end());
    const double value_base = static_cast<double>(vmax) - static_cast<double>(vmin);
    const double abs_eb     = args.eb_rel * value_base;
    const float  ebx2_r     = static_cast<float>(1.0 / (2.0 * abs_eb));
    std::cout << "  value_base=" << value_base << "  abs_eb=" << abs_eb << "\n";

    const uint32_t tile_size = 32 * 8;  // fsz.toml: coder_block_size=32, blocks_per_tile=8
    const size_t   tiles     = (n + tile_size - 1) / tile_size;

    // ── STAGED reference: real QuantizerStage + real AdaptiveLorenzoStage,
    //    driven directly (bypassing Pipeline/DAG) so their raw output buffers
    //    are directly comparable to the fused kernel's. Both classes' execute()
    //    is self-contained given a MemoryPool — no separate finalize() needed.
    MemoryPool pool(MemoryPoolConfig(bytes, 3.0f));

    QuantizerStage<float, uint32_t> qs;
    qs.setErrorBound(args.eb_rel);
    qs.setErrorBoundMode(ErrorBoundMode::NOA);
    qs.setLinearMode(true);

    DevBuf d_codes_s(n * sizeof(uint32_t));
    qs.execute(stream, &pool, {d_raw.ptr}, {d_codes_s.ptr}, {bytes});
    CUDA_CHECK(cudaStreamSynchronize(stream));

    AdaptiveLorenzoStage<int32_t> al;  // defaults: bpt=8, order2=true, centering=true
    DevBuf d_res_s(n * sizeof(int32_t));
    DevBuf d_modes_packed_s((tiles + 3) / 4);
    DevBuf d_means_compact_s(tiles * sizeof(int32_t));  // worst case: every tile centers
    al.execute(stream, &pool, {d_codes_s.ptr},
               {d_res_s.ptr, d_modes_packed_s.ptr, d_means_compact_s.ptr},
               {n * sizeof(int32_t)});
    al.postStreamSync(stream);  // resolves the real compacted means byte count
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto sizes_by_name = al.getActualOutputSizesByName();
    const size_t staged_residuals_bytes = n * sizeof(int32_t);
    const size_t staged_modes_bytes     = (tiles + 3) / 4;
    const size_t staged_means_bytes     = sizes_by_name.count("means") ? sizes_by_name["means"] : 0;
    std::cout << "  staged residuals=" << staged_residuals_bytes
              << "B modes=" << staged_modes_bytes
              << "B means=" << staged_means_bytes << "B\n";

    // ── FUSED path: quant folded into AL's forward kernel ─────────────────────
    DevBuf d_res_f(n * sizeof(int32_t));
    DevBuf d_modes_dense_f(tiles);
    DevBuf d_means_dense_f(tiles * sizeof(int32_t));
    DevBuf d_flags_f((tiles + 1) * sizeof(uint32_t));
    DevBuf d_offsets_f((tiles + 1) * sizeof(uint32_t));
    DevBuf d_modes_packed_f((tiles + 3) / 4);
    DevBuf d_means_compact_f(tiles * sizeof(int32_t));  // worst case: every tile centers

    auto run_fused = [&](cudaStream_t s) {
        probe::launchFusedQuantAL<int32_t>(
            d_raw.as<float>(), ebx2_r, d_res_f.as<int32_t>(), d_modes_dense_f.as<uint8_t>(),
            d_means_dense_f.as<int32_t>(), d_flags_f.as<uint32_t>(), n, tile_size,
            /*enable_order2=*/true, /*enable_centering=*/true,
            EncodingOracleKind::PlainFixedRateBitpack, s);
        CUDA_CHECK(cudaMemsetAsync(d_flags_f.as<uint32_t>() + tiles, 0, sizeof(uint32_t), s));
        size_t cub_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, cub_bytes, d_flags_f.as<uint32_t>(),
                                      d_offsets_f.as<uint32_t>(), tiles + 1, s);
        DevBuf d_cub_tmp(cub_bytes);
        cub::DeviceScan::ExclusiveSum(d_cub_tmp.ptr, cub_bytes, d_flags_f.as<uint32_t>(),
                                      d_offsets_f.as<uint32_t>(), tiles + 1, s);
        fz::launchAdaptiveLorenzoCompact<int32_t>(
            d_modes_dense_f.as<uint8_t>(), d_means_dense_f.as<int32_t>(), d_offsets_f.as<uint32_t>(),
            d_modes_packed_f.as<uint8_t>(), d_means_compact_f.as<int32_t>(), tiles, s);
    };

    run_fused(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint32_t centered_count = 0;
    CUDA_CHECK(cudaMemcpy(&centered_count, d_offsets_f.as<uint32_t>() + tiles,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
    const size_t fused_means_bytes = static_cast<size_t>(centered_count) * sizeof(int32_t);

    // ── Correctness: byte-for-byte comparison ─────────────────────────────────
    bool ok = true;
    {
        std::vector<int32_t> a(n), b(n);
        CUDA_CHECK(cudaMemcpy(a.data(), d_res_s.ptr, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), d_res_f.ptr, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (a != b) { ok = false; std::cerr << "MISMATCH: residuals differ\n"; }
    }
    {
        size_t mb = staged_modes_bytes;
        std::vector<uint8_t> a(mb), b(mb);
        CUDA_CHECK(cudaMemcpy(a.data(), d_modes_packed_s.ptr, mb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), d_modes_packed_f.ptr, mb, cudaMemcpyDeviceToHost));
        if (a != b) { ok = false; std::cerr << "MISMATCH: packed modes differ\n"; }
    }
    if (staged_means_bytes != fused_means_bytes) {
        ok = false;
        std::cerr << "MISMATCH: means size staged=" << staged_means_bytes
                  << " fused=" << fused_means_bytes << "\n";
    } else if (fused_means_bytes > 0) {
        std::vector<int32_t> a(centered_count), b(centered_count);
        CUDA_CHECK(cudaMemcpy(a.data(), d_means_compact_s.ptr, fused_means_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), d_means_compact_f.ptr, fused_means_bytes, cudaMemcpyDeviceToHost));
        if (a != b) { ok = false; std::cerr << "MISMATCH: compacted means differ\n"; }
    }
    std::cout << "Correctness (fused vs staged, byte-for-byte): " << (ok ? "PASS" : "FAIL") << "\n";
    if (!ok) return 1;

    // ── Timing: fused kernel + compaction, warmup + N reps, median ────────────
    const int warmup = 5;
    for (int i = 0; i < warmup; ++i) run_fused(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> times_ms(args.reps);
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    for (int i = 0; i < args.reps; ++i) {
        CUDA_CHECK(cudaEventRecord(t0, stream));
        run_fused(stream);
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&times_ms[i], t0, t1));
    }
    std::sort(times_ms.begin(), times_ms.end());
    const float median_ms = times_ms[times_ms.size() / 2];
    const double gbs = (bytes / 1e9) / (median_ms / 1e3);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nFUSED quant+AdaptiveLorenzo (median of " << args.reps << "): "
              << median_ms << " ms  (" << gbs << " GB/s, input-relative)\n";
    std::cout << "Compare against fzgmod-cli --profile's Quantizer+AdaptiveLorenzo\n"
              << "stage-time SUM for the same file/eb to see the deleted-pass effect.\n";
    return 0;
}
