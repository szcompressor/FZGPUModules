/**
 * profiling/decompress_concurrency.cu
 *
 * Measures whether decompress() calls on independent Pipeline instances and
 * independent CUDA streams actually run CONCURRENTLY on the GPU — the property a
 * double/triple-buffered staging pipeline (or a decompressor running alongside a
 * scientific simulator) needs to overlap PCIe transfer with decode and approach
 * the roofline.
 *
 * Each "slot" is its own Pipeline instance (PREALLOCATE), its own stream, its own
 * device buffers, and is driven by its own host thread. The harness uses
 * decompressInto() — the fully stream-asynchronous, caller-buffer decode that does
 * NO cudaMalloc/cudaFree/device-wide cudaStreamSynchronize internally — so the only
 * remaining synchronization is each thread's own per-iteration stream sync, which
 * stalls only the calling thread.
 *
 * Reports:
 *   serial_ms      — K*iters decodes issued back-to-back from ONE thread
 *   concurrent_ms  — K threads each doing `iters` decodes on their own slot
 *   speedup        — serial_ms / concurrent_ms  (→ K means full overlap;
 *                    → 1 means something is serializing the device)
 *
 * Pipelines are selectable (last arg): `rle` (LorenzoQuant->RLE, the clean
 * baseline), `huffman` (cuSZ-style LorenzoQuant->Huffman) and `rze` (PFPL-style
 * Quantizer->Bitshuffle->RZE). All three inverse coders now read their headers via
 * async-on-stream + a stream-scoped sync (no device-wide barrier), so all three
 * overlap; `rle` is the reference since it never had a barrier.
 *
 * Usage:
 *   ./build/release/bin/profiling/decompress_concurrency [num_slots] [iters] [block_floats] [pipeline]
 *   defaults: num_slots=4  iters=200  block_floats=1048576 (4 MB)  pipeline=rle
 *   pipeline ∈ { rle         : LorenzoQuant -> RLE                    (clean baseline)
 *                huffman     : LorenzoQuant -> Huffman               (cuSZ-style)
 *                rze         : Quantizer -> Bitshuffle -> RZE        (PFPL-style)
 *                bitplane_rze: LorenzoQuant -> BitplaneRZE           (FZ-GPU lossless) }
 *
 * Build:
 *   cmake --preset release -DBUILD_PROFILING=ON && cmake --build build/release -j$(nproc)
 *   Binary: build/release/bin/profiling/decompress_concurrency
 */

#include "fzgpumodules.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace fz;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// One independent decode slot: its own pipeline, stream, blob, and output buffer.
struct Slot {
    std::unique_ptr<Pipeline> pipe;
    cudaStream_t stream   = nullptr;
    void*        d_blob   = nullptr;   // independent copy of the compressed payload
    size_t       blob_sz  = 0;
    void*        d_out    = nullptr;   // caller-owned decode output (PREALLOCATE-friendly)
    size_t       out_cap  = 0;
    size_t       in_bytes = 0;
};

static void build_pipeline(Pipeline& p, float eb, const std::string& kind) {
    if (kind == "huffman") {
        // cuSZ-style: LorenzoQuant -> Huffman (inverse Huffman header read).
        auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
        lq->setErrorBound(eb);
        lq->setErrorBoundMode(ErrorBoundMode::ABS);
        lq->setQuantRadius(512);
        lq->setOutlierCapacity(0.10f);
        lq->setZigzagCodes(true);
        auto* huf = p.addStage<HuffmanStage<uint16_t>>();
        huf->setBklen(1024);
        p.connect(huf, lq, "codes");
    } else if (kind == "rze") {
        // PFPL-style: Quantizer -> Bitshuffle -> RZE (inverse RZE header read).
        auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
        q->setErrorBound(eb);
        q->setErrorBoundMode(ErrorBoundMode::ABS);
        q->setQuantRadius(512);
        auto* bshuf = p.addStage<BitshuffleStage>();
        bshuf->setElementWidth(4);
        auto* rze = p.addStage<RZEStage>();
        p.connect(bshuf, q, "codes");
        p.connect(rze, bshuf);
    } else if (kind == "bitplane_rze") {
        // FZ-GPU lossless: LorenzoQuant -> BitplaneRZE (inverse header read now
        // stream-scoped, so this pipeline also overlaps across slots).
        auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
        lq->setErrorBound(eb);
        lq->setErrorBoundMode(ErrorBoundMode::ABS);
        lq->setQuantRadius(512);
        lq->setOutlierCapacity(0.10f);
        lq->setZigzagCodes(true);
        auto* bprze = p.addStage<BitplaneRZEStage>();
        p.connect(bprze, lq, "codes");
    } else {  // "rle"
        auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
        lq->setErrorBound(eb);
        lq->setErrorBoundMode(ErrorBoundMode::ABS);
        lq->setQuantRadius(512);
        lq->setOutlierCapacity(0.10f);
        auto* rle = p.addStage<RLEStage<uint16_t>>();
        p.connect(rle, lq, "codes");
    }
    p.finalize();
}

int main(int argc, char** argv) {
    const int         K     = (argc > 1) ? std::atoi(argv[1]) : 4;
    const int         ITERS = (argc > 2) ? std::atoi(argv[2]) : 200;
    const size_t      N     = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : (1u << 20);
    const std::string KIND  = (argc > 4) ? argv[4] : "rle";
    const float       EB    = 1e-3f;
    const size_t in_bytes = N * sizeof(float);

    std::printf("decompress_concurrency: pipeline=%s slots=%d iters=%d block=%zu floats (%.1f MB)\n",
                KIND.c_str(), K, ITERS, N, in_bytes / 1048576.0);

    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
    std::printf("GPU: %s  (async engines=%d, MPs=%d)\n\n",
                prop.name, prop.asyncEngineCount, prop.multiProcessorCount);

    // Synthetic, mildly spiky data so RLE has real work to do.
    std::vector<float> host(N);
    for (size_t i = 0; i < N; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(N);
        host[i] = std::sin(50.0f * t) + 0.5f * std::cos(13.0f * t)
                + ((i % 521 == 0) ? 8.0f : 0.0f);
    }

    // ── Per-slot setup (outside any timing) ───────────────────────────────────
    std::vector<Slot> slots(K);
    for (int k = 0; k < K; ++k) {
        Slot& s = slots[k];
        s.in_bytes = in_bytes;
        cudaStreamCreate(&s.stream);

        s.pipe = std::make_unique<Pipeline>(in_bytes, MemoryStrategy::PREALLOCATE);
        build_pipeline(*s.pipe, EB, KIND);

        float* d_in = nullptr;
        cudaMalloc(&d_in, in_bytes);
        cudaMemcpy(d_in, host.data(), in_bytes, cudaMemcpyHostToDevice);

        void*  d_comp = nullptr; size_t comp_sz = 0;
        s.pipe->compress(d_in, in_bytes, &d_comp, &comp_sz, s.stream);
        cudaStreamSynchronize(s.stream);

        // Independent copy of the blob (compress output is pool-owned).
        s.blob_sz = comp_sz;
        cudaMalloc(&s.d_blob, comp_sz);
        cudaMemcpy(s.d_blob, d_comp, comp_sz, cudaMemcpyDeviceToDevice);

        // Caller-owned decode output buffer.
        s.out_cap = in_bytes;
        cudaMalloc(&s.d_out, s.out_cap);

        cudaFree(d_in);

        // Warm up the inverse DAG once (untimed): the first decode builds + sizes
        // the inverse buffers; the timed loop then runs steady-state.
        size_t actual = 0;
        s.pipe->decompressInto(s.d_blob, s.blob_sz, s.d_out, s.out_cap, &actual, s.stream);
        cudaStreamSynchronize(s.stream);

        if (actual != in_bytes) {
            std::printf("  slot %d: WARN decode size %zu != %zu\n", k, actual, in_bytes);
        }
    }
    cudaDeviceSynchronize();

    auto one_slot_loop = [](Slot& s, int iters) {
        for (int i = 0; i < iters; ++i) {
            size_t actual = 0;
            s.pipe->decompressInto(s.d_blob, s.blob_sz, s.d_out, s.out_cap, &actual, s.stream);
            cudaStreamSynchronize(s.stream);   // stream-scoped: stalls only this thread
        }
    };

    // ── Serial baseline: all K*ITERS decodes from ONE thread ──────────────────
    cudaDeviceSynchronize();
    auto t0 = Clock::now();
    for (int k = 0; k < K; ++k) one_slot_loop(slots[k], ITERS);
    cudaDeviceSynchronize();
    const double serial_ms = ms_since(t0);

    // ── Concurrent: K threads, each driving its own slot/stream ───────────────
    cudaDeviceSynchronize();
    t0 = Clock::now();
    {
        std::vector<std::thread> threads;
        threads.reserve(K);
        for (int k = 0; k < K; ++k)
            threads.emplace_back([&, k]() { one_slot_loop(slots[k], ITERS); });
        for (auto& th : threads) th.join();
    }
    cudaDeviceSynchronize();
    const double concurrent_ms = ms_since(t0);

    const double speedup       = serial_ms / concurrent_ms;
    const double total_bytes   = static_cast<double>(in_bytes) * K * ITERS;
    const double serial_gbs    = total_bytes / (serial_ms     * 1e6);
    const double conc_gbs      = total_bytes / (concurrent_ms * 1e6);

    std::printf("\n── Results (%d decodes each) ──────────────────────────────\n",
                K * ITERS);
    std::printf("  serial     : %8.2f ms   %6.1f GB/s\n", serial_ms,     serial_gbs);
    std::printf("  concurrent : %8.2f ms   %6.1f GB/s\n", concurrent_ms, conc_gbs);
    std::printf("  speedup    : %6.2fx  (ideal ~%d; >1 means real GPU overlap)\n",
                speedup, K);

    for (auto& s : slots) {
        if (s.d_blob) cudaFree(s.d_blob);
        if (s.d_out)  cudaFree(s.d_out);
        s.pipe.reset();
        if (s.stream) cudaStreamDestroy(s.stream);
    }
    return 0;
}
