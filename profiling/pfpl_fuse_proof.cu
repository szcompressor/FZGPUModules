// Go/no-go + generalization proof for the chunk-cooperative fusion HARNESS.
//
// The harness (modules/fused/chunk_fusion/chunk_fusion.cuh) composes per-stage
// __device__ ops into one CTA-per-chunk kernel. This proves it is (a) byte-
// identical to the staged pipeline and (b) GENERALIZABLE across the coder axis:
// the SAME harness, given a different coder op, fuses a PFPL-with-RRE pipeline
// with no new glue. Runs the whole thing for RZE and RRE.
//
// PFPL = Quantizer(NOA,inplace,zigzag) -> Difference(negabinary) -> Bitshuffle(ew4)
//        -> <coder>. Restriction: inplace-outlier quant fast path (single port).

#include "fzgpumodules.h"
#include "quantizers/quantizer/quantizer.h"
#include "predictors/diff/diff.h"
#include "shufflers/bitshuffle/bitshuffle_stage.h"
#include "coders/rze/rze_stage.h"
#include "coders/rre/rre_stage.h"
#include "fused/chunk_fusion/chunk_fusion.h"
#include "mem/mempool.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace fz;
using byte  = uint8_t;
using Clock = std::chrono::high_resolution_clock;

static const char*      DATA_PATH = FZ_DATA_DIR "/CLDHGH.f32";
static constexpr size_t DIM_X = 3600, DIM_Y = 1800;
static constexpr int    CS    = 16384, NELEM = CS / 4;

static double ms(std::chrono::time_point<Clock> a, std::chrono::time_point<Clock> b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static double gbs(size_t bytes, double milli) { return bytes / (milli * 1e-3) / 1e9; }

// Manual staged 4-stage forward (inplace quant, single "codes" port) -> raw coder
// archive. CoderStage is RZEStage or RREStage (same interface). Returns archive bytes.
template<class CoderStage>
static size_t staged_forward(const float* d_in, size_t n, size_t in_bytes, float eb,
                             float value_base, MemoryPool& pool, cudaStream_t stream,
                             std::vector<byte>& out) {
    uint32_t* d_codes; cudaMalloc(&d_codes, n * 4);
    uint32_t* d_nb;    cudaMalloc(&d_nb, n * 4);
    uint32_t* d_bs;    cudaMalloc(&d_bs, n * 4);
    byte*     d_arch;  cudaMalloc(&d_arch, in_bytes + 4096);

    QuantizerStage<float, uint32_t> q;
    q.setErrorBound(eb); q.setErrorBoundMode(ErrorBoundMode::NOA);
    q.setValueBase(value_base); q.setQuantRadius(32768);
    q.setZigzagCodes(true); q.setInplaceOutliers(true);   // single-port -> fusable
    DifferenceStage<int32_t, uint32_t> df; df.setChunkSize(CS);
    BitshuffleStage bs; bs.setElementWidth(4); bs.setBlockSize(CS);
    CoderStage rz; rz.setWordSize(1); rz.setChunkSize(CS);

    { std::vector<void*> i{(void*)d_in}, o{d_codes}; std::vector<size_t> s{in_bytes};
      q.execute(stream, &pool, i, o, s); }
    { std::vector<void*> i{d_codes}, o{d_nb}; std::vector<size_t> s{n*4};
      df.execute(stream, &pool, i, o, s); }
    { std::vector<void*> i{d_nb}, o{d_bs}; std::vector<size_t> s{n*4};
      bs.execute(stream, &pool, i, o, s); }
    { std::vector<void*> i{d_bs}, o{d_arch}; std::vector<size_t> s{n*4};
      rz.execute(stream, &pool, i, o, s); rz.postStreamSync(stream); }
    cudaDeviceSynchronize();

    const size_t bytes = rz.getActualOutputSize(0);
    out.resize(bytes);
    cudaMemcpy(out.data(), d_arch, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_codes); cudaFree(d_nb); cudaFree(d_bs); cudaFree(d_arch);
    return bytes;
}

int main(int argc, char** argv) {
    const int   runs = argc > 1 ? atoi(argv[1]) : 15;
    const float eb   = argc > 2 ? (float)atof(argv[2]) : 1e-4f;

    size_t n_full = DIM_X * DIM_Y;
    std::vector<float> h(n_full);
    { FILE* fp = fopen(DATA_PATH, "rb");
      if (!fp) { printf("cannot open %s\n", DATA_PATH); return 1; }
      if (fread(h.data(), 4, n_full, fp) != n_full) { printf("short read\n"); return 1; }
      fclose(fp); }
    const size_t n = (n_full / NELEM) * NELEM;   // whole chunks only (tail handled but keep proof clean)
    h.resize(n);
    const size_t in_bytes = n * sizeof(float);

    float vmin = h[0], vmax = h[0];
    for (float v : h) { vmin = std::min(vmin, v); vmax = std::max(vmax, v); }
    const float value_base = vmax - vmin;
    const float abs_eb = eb * value_base;
    const float ebx2_r = 1.0f / (2.0f * abs_eb);
    const float INF = std::numeric_limits<float>::infinity();

    printf("Chunk-fusion harness proof — CLDHGH  n=%zu (%zu chunks)  eb=%.1e NOA\n",
           n, n / NELEM, (double)eb);
    printf("value_base=%.6g  ebx2_r=%.6g\n\n", value_base, ebx2_r);

    float* d_in; cudaMalloc(&d_in, in_bytes);
    cudaMemcpy(d_in, h.data(), in_bytes, cudaMemcpyHostToDevice);
    MemoryPool pool(MemoryPoolConfig{});
    cudaStream_t stream = 0;
    byte* d_arch; cudaMalloc(&d_arch, in_bytes + 4096);

    int failures = 0;
    auto run_case = [&](const char* name, fused::ChunkCoderKind coder, auto staged_fn) {
        std::vector<byte> A_staged;
        const size_t sbytes = staged_fn(A_staged);

        auto fused_once = [&]() -> size_t {
            return fused::launchFusedChunkPfpl(coder, d_in, n, ebx2_r, 32768, INF, d_arch, &pool, stream);
        };
        const size_t fbytes = fused_once();
        cudaDeviceSynchronize();
        std::vector<byte> A_fused(fbytes);
        cudaMemcpy(A_fused.data(), d_arch, fbytes, cudaMemcpyDeviceToHost);

        const bool ok = (fbytes == sbytes) && (memcmp(A_fused.data(), A_staged.data(), fbytes) == 0);
        if (!ok) failures++;

        double fused_ms = 1e30;
        for (int r = 0; r < runs; ++r) { auto t0 = Clock::now(); fused_once(); cudaDeviceSynchronize();
            fused_ms = std::min(fused_ms, ms(t0, Clock::now())); }

        printf("[%s]  staged=%.3f MB  fused=%.3f MB (%.2fx)  byte-identical: %s   fused %.3f ms (%.0f GB/s)\n",
               name, sbytes/1048576.0, fbytes/1048576.0, (double)in_bytes/fbytes,
               ok ? "YES" : "NO [FAIL]", fused_ms, gbs(in_bytes, fused_ms));
    };

    run_case("RZE", fused::ChunkCoderKind::RZE, [&](std::vector<byte>& o){
        return staged_forward<RZEStage>(d_in, n, in_bytes, eb, value_base, pool, stream, o); });
    run_case("RRE", fused::ChunkCoderKind::RRE, [&](std::vector<byte>& o){
        return staged_forward<RREStage>(d_in, n, in_bytes, eb, value_base, pool, stream, o); });

    printf("\n%s — the SAME harness fused both coders with no new glue.\n",
           failures == 0 ? "GENERALIZATION OK" : "FAILURE");
    cudaFree(d_in); cudaFree(d_arch);
    return failures ? 2 : 0;
}
