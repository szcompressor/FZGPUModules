/**
 * examples/decode_only_slots.cpp
 *
 * Decode external in-memory blobs WITHOUT a warmup compress().
 *
 * Motivating use case: a streaming / double-buffered decode loop (e.g. K slots
 * pulling independently-compressed blocks off disk and decompressing them on the
 * GPU). Each slot is a Pipeline instance that ONLY ever decompresses — it never
 * compresses anything itself.
 *
 * The in-memory decompress() path normally depends on state left behind by a
 * preceding compress() on the same instance: the archive layout AND the
 * data-dependent inverse metadata (HuffmanStage's symbol count, the quantizer
 * outlier count — which changes block to block and is NOT in the raw blob). The
 * old workaround was to run a throwaway compress() over dummy data on each slot
 * just to populate that state. This example shows the proper replacement:
 *
 *   Producer (once, per block, at compress time):
 *     std::vector<uint8_t> header = p.serializeHeaderToMemory();
 *     // store `header` alongside the compress() output blob
 *
 *   Consumer (a fresh, finalized, NEVER-compressed pipeline of the same topology):
 *     slot.decompressFromMemory(header.data(), header.size(),
 *                               d_blob, blob_size, &d_out, &out_sz, stream);
 *
 * decompressFromMemory() = primeInverseFromHeader() + decompress() fused into one
 * call. It restores the per-block metadata from the header and decodes the blob,
 * reusing the slot's cached inverse DAG across blocks (no per-block DAG rebuild).
 *
 * No external data files required — uses synthetic float data.
 *
 * Usage:
 *   ./build/release/bin/examples/decode_only_slots
 *
 * Build:
 *   cmake --preset release -DBUILD_EXAMPLES=ON && cmake --build build/release -j$(nproc)
 *   Binary: build/release/bin/examples/decode_only_slots
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace fz;

static constexpr size_t BLOCK_N     = 1 << 16;        // 64K floats per "row-block"
static constexpr size_t BLOCK_BYTES = BLOCK_N * sizeof(float);
static constexpr float  ERROR_BOUND = 1e-3f;
static constexpr int    NUM_BLOCKS  = 4;

// The cuSZ-style pipeline: LorenzoQuant (lossy float predictor + quantizer, emits
// outliers) → Huffman (entropy coder needing a symbol count). Both carry inverse
// metadata that lives only in the header, not the raw blob — which is exactly why
// a decode-only slot needs serializeHeaderToMemory()/primeInverseFromHeader().
static void build_pipeline(Pipeline& p) {
    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(ERROR_BOUND);
    lq->setErrorBoundMode(ErrorBoundMode::ABS);
    lq->setQuantRadius(512);
    lq->setOutlierCapacity(0.10f);
    lq->setZigzagCodes(true);              // keep codes in [0, bklen) for Huffman

    auto* huf = p.addStage<HuffmanStage<uint16_t>>();
    huf->setBklen(1024);
    p.connect(huf, lq, "codes");

    p.finalize();
}

// Distinct synthetic block: a sinusoid plus periodic spikes, so each block has a
// DIFFERENT outlier count (the metadata that must travel in the header).
static std::vector<float> make_block(int seed) {
    std::vector<float> h(BLOCK_N);
    const float phase = 0.3f * static_cast<float>(seed);
    const int   spike = 257 + 13 * seed;   // varies the spike density per block
    for (size_t i = 0; i < BLOCK_N; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(BLOCK_N);
        float v = std::sin(6.2831853f * t + phase) + 0.5f * std::cos(18.8f * t);
        if (static_cast<int>(i) % spike == 0) v += 25.0f;   // outlier-inducing spike
        h[i] = v;
    }
    return h;
}

// One compressed block as it would sit on disk / in a ring buffer: the metadata
// header (host bytes) + an independent device copy of the compressed payload.
struct StoredBlock {
    std::vector<uint8_t> header;
    void*                d_blob   = nullptr;
    size_t               blob_size = 0;
    std::vector<float>   original;   // kept only so the example can verify
};

// PRODUCER: compress a block and capture (header, blob). In a real system the
// producer is a separate process/run; here we just drop the Pipeline immediately
// afterwards to prove the consumer shares no state with it.
static StoredBlock produce_block(int seed) {
    StoredBlock out;
    out.original = make_block(seed);

    float* d_in = nullptr;
    cudaMalloc(&d_in, BLOCK_BYTES);
    cudaMemcpy(d_in, out.original.data(), BLOCK_BYTES, cudaMemcpyHostToDevice);

    Pipeline prod(BLOCK_BYTES);
    build_pipeline(prod);

    fz::BorrowedDeviceBuffer comp_buf = prod.compress({d_in, BLOCK_BYTES});
    cudaDeviceSynchronize();

    // (1) the small metadata header — store this next to the blob.
    out.header = prod.serializeHeaderToMemory();

    // (2) an independent copy of the compressed payload (compress() output is
    //     pool-owned and dies with `prod`).
    out.blob_size = comp_buf.bytes();
    cudaMalloc(&out.d_blob, comp_buf.bytes());
    cudaMemcpy(out.d_blob, comp_buf.data(), comp_buf.bytes(), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    return out;   // `prod` (and its pool) destroyed here — blob stands alone
}

static float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float e = 0.0f;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) e = std::max(e, std::fabs(a[i] - b[i]));
    return e;
}

int main() {
    std::printf("── Producer: compress %d blocks, keep (header + blob) per block ──\n",
                NUM_BLOCKS);
    std::vector<StoredBlock> blocks;
    blocks.reserve(NUM_BLOCKS);
    for (int s = 0; s < NUM_BLOCKS; ++s) {
        blocks.push_back(produce_block(s));
        std::printf("  block %d: header=%zu B  blob=%zu B  ratio=%.2fx\n",
                    s, blocks[s].header.size(), blocks[s].blob_size,
                    static_cast<double>(BLOCK_BYTES) / static_cast<double>(blocks[s].blob_size));
    }

    // ── Consumer: ONE decode-only slot, never compress()ed ────────────────────
    std::printf("\n── Consumer: one decode-only slot, NO warmup compress() ──────────\n");
    Pipeline slot(BLOCK_BYTES);
    build_pipeline(slot);                       // same topology as the producer
    slot.setPoolManagedDecompOutput(true);      // pool-owned output (default)

    bool all_ok = true;
    for (int s = 0; s < NUM_BLOCKS; ++s) {
        const StoredBlock& blk = blocks[s];

        void*  d_out = nullptr;
        size_t out_sz = 0;
        // Single per-block call: restore this block's metadata from its header and
        // decode the blob, reusing the slot's cached inverse DAG.
        slot.decompressFromMemory(blk.header.data(), blk.header.size(),
                                  blk.d_blob, blk.blob_size, &d_out, &out_sz);
        cudaDeviceSynchronize();

        std::vector<float> recon(out_sz / sizeof(float));
        cudaMemcpy(recon.data(), d_out, out_sz, cudaMemcpyDeviceToHost);
        // d_out is pool-owned by `slot` — do NOT cudaFree it.

        const float err = max_abs_error(blk.original, recon);
        const bool  ok  = (out_sz == BLOCK_BYTES) && (err <= ERROR_BOUND * 1.01f);
        all_ok &= ok;
        std::printf("  decoded block %d: out=%zu B  max_abs_error=%.3e  %s\n",
                    s, out_sz, err, ok ? "PASS" : "FAIL");
    }

    for (auto& blk : blocks) cudaFree(blk.d_blob);

    std::printf("\n%s\n", all_ok ? "All blocks decoded correctly with no warmup compress()."
                                 : "FAILED");
    return all_ok ? 0 : 1;
}
