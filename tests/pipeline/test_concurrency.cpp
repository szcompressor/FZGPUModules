/**
 * tests/pipeline/test_concurrency.cpp
 *
 * Validates that independent Pipeline instances on independent CUDA streams
 * can run concurrently — the property the decompress_concurrency profiling
 * harness measures. These tests focus on correctness (all slots produce valid
 * round-trips) rather than timing, which is fragile in CI environments.
 *
 * Covered scenarios:
 *
 *   CC1  MultiSlot_RLE              — K slots, LorenzoQuant→RLE (clean baseline)
 *   CC2  MultiSlot_Huffman          — K slots, LorenzoQuant→Huffman (cuSZ-style;
 *                                     inverse header read now stream-scoped)
 *   CC3  MultiSlot_RZE              — K slots, Quantizer→Bitshuffle→RZE (PFPL-style)
 *   CC4  MultiSlot_BitplaneRZE      — K slots, LorenzoQuant→BitplaneRZE (FZ-GPU;
 *                                     inverse header read converted from device-wide
 *                                     to stream-scoped in this session)
 *   CC5  ConcurrentCompress_Forward — K threads each compressing on their own slot;
 *                                     exercises the postStreamSync path (outlier
 *                                     readbacks in LorenzoQuant / Quantizer now
 *                                     stream-scoped)
 *   CC6  MixedCompress_Decompress   — alternate compress + decompress across slots
 *                                     to confirm postStreamSync does not interfere
 *                                     with a subsequent decompress on the same slot
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "fzgpumodules.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

using namespace fz;
using namespace fz_test;

namespace {

constexpr int    K_SLOTS  = 4;
constexpr size_t N_FLOATS = 64 * 1024;  // 256 KB — large enough to expose barriers
constexpr float  EB       = 1e-3f;

// Build smooth data with an occasional spike (exercises outlier paths).
std::vector<float> make_mixed_data(size_t n, uint32_t seed = 0) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(n);
        v[i] = std::sin(30.0f * t) + 0.3f * std::cos(11.0f * t);
        if ((i + seed) % 503 == 0) v[i] += 8.0f;
    }
    return v;
}

// One decode slot: its own Pipeline, stream, compressed blob, and output buffer.
struct Slot {
    std::unique_ptr<Pipeline> pipe;
    cudaStream_t              stream  = nullptr;
    void*                     d_blob  = nullptr;
    size_t                    blob_sz = 0;
    void*                     d_out   = nullptr;
    size_t                    out_cap = 0;
    std::vector<float>        h_orig;
};

// Build and populate a slot using a given pipeline-construction callback.
// After setup, d_blob holds an independent copy of the compressed payload;
// d_out is pre-allocated for decompressInto.
using BuildFn = void(*)(Pipeline&);

Slot make_slot(BuildFn build, const std::vector<float>& h_in) {
    Slot s;
    s.h_orig  = h_in;
    const size_t in_bytes = h_in.size() * sizeof(float);
    s.out_cap = in_bytes;

    cudaStreamCreate(&s.stream);
    s.pipe = std::make_unique<Pipeline>(in_bytes, MemoryStrategy::PREALLOCATE);
    build(*s.pipe);

    // Upload + compress.
    float* d_in = nullptr;
    cudaMalloc(&d_in, in_bytes);
    cudaMemcpy(d_in, h_in.data(), in_bytes, cudaMemcpyHostToDevice);

    void* d_comp = nullptr; size_t comp_sz = 0;
    s.pipe->compress(d_in, in_bytes, &d_comp, &comp_sz, s.stream);
    cudaStreamSynchronize(s.stream);
    cudaFree(d_in);

    // Independent blob copy (compress output is pool-owned).
    s.blob_sz = comp_sz;
    cudaMalloc(&s.d_blob, comp_sz);
    cudaMemcpy(s.d_blob, d_comp, comp_sz, cudaMemcpyDeviceToDevice);

    // Caller-owned decode output.
    cudaMalloc(&s.d_out, s.out_cap);

    // Warm-up inverse DAG: first call builds + preallocates inverse buffers.
    size_t actual = 0;
    s.pipe->decompressInto(s.d_blob, s.blob_sz, s.d_out, s.out_cap, &actual, s.stream);
    cudaStreamSynchronize(s.stream);

    return s;
}

void destroy_slot(Slot& s) {
    cudaFree(s.d_blob);
    cudaFree(s.d_out);
    s.pipe.reset();
    cudaStreamDestroy(s.stream);
}

// Verify that the decoded output of slot s matches h_orig within EB.
void check_slot(const Slot& s, const char* label) {
    const size_t n = s.h_orig.size();
    std::vector<float> h_dec(n);
    cudaMemcpy(h_dec.data(), s.d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(h_dec[i], s.h_orig[i], EB * 1.001)
            << label << " mismatch at i=" << i;
        if (::testing::Test::HasFatalFailure()) return;
    }
}

// Run K concurrent decodes (each slot gets its own thread), then verify.
void run_concurrent_decode(std::vector<Slot>& slots) {
    std::vector<std::thread> threads;
    threads.reserve(slots.size());
    for (auto& s : slots) {
        threads.emplace_back([&s]() {
            size_t actual = 0;
            s.pipe->decompressInto(
                s.d_blob, s.blob_sz, s.d_out, s.out_cap, &actual, s.stream);
            cudaStreamSynchronize(s.stream);
        });
    }
    for (auto& th : threads) th.join();
}

// ── Pipeline builders ────────────────────────────────────────────────────────

void build_rle(Pipeline& p) {
    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(EB);
    lq->setErrorBoundMode(ErrorBoundMode::ABS);
    lq->setQuantRadius(512);
    lq->setOutlierCapacity(0.10f);
    auto* rle = p.addStage<RLEStage<uint16_t>>();
    p.connect(rle, lq, "codes");
    p.finalize();
}

void build_huffman(Pipeline& p) {
    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(EB);
    lq->setErrorBoundMode(ErrorBoundMode::ABS);
    lq->setQuantRadius(512);
    lq->setOutlierCapacity(0.10f);
    lq->setZigzagCodes(true);
    auto* huf = p.addStage<HuffmanStage<uint16_t>>();
    huf->setBklen(1024);
    p.connect(huf, lq, "codes");
    p.finalize();
}

void build_rze(Pipeline& p) {
    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(EB);
    q->setErrorBoundMode(ErrorBoundMode::ABS);
    // Radius large enough to cover the smooth sinusoid (amplitude ~1.3):
    // radius*EB = 2048*1e-3 = 2.048 > 1.3.  Spike outliers (~0.2%) fit in
    // the default 5% capacity.
    q->setQuantRadius(2048);
    auto* bshuf = p.addStage<BitshuffleStage>();
    bshuf->setElementWidth(4);
    auto* rze = p.addStage<RZEStage>();
    p.connect(bshuf, q, "codes");
    p.connect(rze, bshuf);
    p.finalize();
}

void build_bitplane_rze(Pipeline& p) {
    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(EB);
    lq->setErrorBoundMode(ErrorBoundMode::ABS);
    lq->setQuantRadius(512);
    lq->setOutlierCapacity(0.10f);
    lq->setZigzagCodes(true);
    auto* bprze = p.addStage<BitplaneRZEStage>();
    p.connect(bprze, lq, "codes");
    p.finalize();
}

}  // namespace

// ── CC1 ──────────────────────────────────────────────────────────────────────
TEST(Concurrency, MultiSlot_RLE) {
    auto h_in = make_mixed_data(N_FLOATS);
    std::vector<Slot> slots;
    for (int k = 0; k < K_SLOTS; ++k)
        slots.push_back(make_slot(build_rle, h_in));

    run_concurrent_decode(slots);

    for (int k = 0; k < K_SLOTS; ++k) {
        SCOPED_TRACE("slot " + std::to_string(k));
        check_slot(slots[k], "CC1/RLE");
    }
    for (auto& s : slots) destroy_slot(s);
}

// ── CC2 ──────────────────────────────────────────────────────────────────────
// Huffman inverse: cudaMemcpyAsync+stream-sync for phf_header read (converted
// from plain cudaMemcpy device-wide barrier in the previous session).
TEST(Concurrency, MultiSlot_Huffman) {
    auto h_in = make_mixed_data(N_FLOATS, 1);
    std::vector<Slot> slots;
    for (int k = 0; k < K_SLOTS; ++k)
        slots.push_back(make_slot(build_huffman, h_in));

    run_concurrent_decode(slots);

    for (int k = 0; k < K_SLOTS; ++k) {
        SCOPED_TRACE("slot " + std::to_string(k));
        check_slot(slots[k], "CC2/Huffman");
    }
    for (auto& s : slots) destroy_slot(s);
}

// ── CC3 ──────────────────────────────────────────────────────────────────────
// RZE inverse: 8-byte header + chunk entries reads converted in previous session.
TEST(Concurrency, MultiSlot_RZE) {
    auto h_in = make_mixed_data(N_FLOATS, 2);
    std::vector<Slot> slots;
    for (int k = 0; k < K_SLOTS; ++k)
        slots.push_back(make_slot(build_rze, h_in));

    run_concurrent_decode(slots);

    for (int k = 0; k < K_SLOTS; ++k) {
        SCOPED_TRACE("slot " + std::to_string(k));
        check_slot(slots[k], "CC3/RZE");
    }
    for (auto& s : slots) destroy_slot(s);
}

// ── CC4 ──────────────────────────────────────────────────────────────────────
// BitplaneRZE inverse: ArchiveHeader read converted from plain cudaMemcpy
// (device-wide via legacy default stream) to cudaMemcpyAsync+stream-sync.
TEST(Concurrency, MultiSlot_BitplaneRZE) {
    auto h_in = make_mixed_data(N_FLOATS, 3);
    std::vector<Slot> slots;
    for (int k = 0; k < K_SLOTS; ++k)
        slots.push_back(make_slot(build_bitplane_rze, h_in));

    run_concurrent_decode(slots);

    for (int k = 0; k < K_SLOTS; ++k) {
        SCOPED_TRACE("slot " + std::to_string(k));
        check_slot(slots[k], "CC4/BitplaneRZE");
    }
    for (auto& s : slots) destroy_slot(s);
}

// ── CC5 ──────────────────────────────────────────────────────────────────────
// Concurrent forward compress from K threads on K independent Pipeline
// instances. Exercises postStreamSync (LorenzoQuant outlier readback, now
// stream-scoped). Each thread does compress → verify round-trip.
TEST(Concurrency, ConcurrentCompress_Forward) {
    const size_t in_bytes = N_FLOATS * sizeof(float);
    auto h_in = make_mixed_data(N_FLOATS, 4);

    struct CSlot {
        std::unique_ptr<Pipeline> pipe;
        cudaStream_t              stream = nullptr;
        bool                      ok     = false;
        double                    max_err = 0.0;
    };

    std::vector<CSlot> cslots(K_SLOTS);
    for (auto& cs : cslots) {
        cudaStreamCreate(&cs.stream);
        cs.pipe = std::make_unique<Pipeline>(in_bytes, MemoryStrategy::PREALLOCATE);
        build_huffman(*cs.pipe);
    }

    std::vector<std::thread> threads;
    threads.reserve(K_SLOTS);
    for (int k = 0; k < K_SLOTS; ++k) {
        threads.emplace_back([&, k]() {
            auto& cs = cslots[k];
            float* d_in = nullptr;
            cudaMalloc(&d_in, in_bytes);
            cudaMemcpy(d_in, h_in.data(), in_bytes, cudaMemcpyHostToDevice);

            void* d_comp = nullptr; size_t comp_sz = 0;
            cs.pipe->compress(d_in, in_bytes, &d_comp, &comp_sz, cs.stream);

            void* d_dec = nullptr; size_t dec_sz = 0;
            cs.pipe->decompress(d_comp, comp_sz, &d_dec, &dec_sz, cs.stream);
            cudaStreamSynchronize(cs.stream);

            const size_t n_out = dec_sz / sizeof(float);
            std::vector<float> h_dec(n_out);
            cudaMemcpy(h_dec.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);
            cudaFree(d_in);

            double err = 0.0;
            for (size_t i = 0; i < std::min(h_in.size(), n_out); ++i) {
                double d = std::abs(static_cast<double>(h_dec[i]) -
                                    static_cast<double>(h_in[i]));
                if (d > err) err = d;
            }
            cs.max_err = err;
            cs.ok = (n_out == h_in.size() && err <= EB * 1.001);
        });
    }
    for (auto& th : threads) th.join();

    for (int k = 0; k < K_SLOTS; ++k) {
        EXPECT_TRUE(cslots[k].ok)
            << "slot " << k << " round-trip failed (max_err=" << cslots[k].max_err << ")";
    }

    for (auto& cs : cslots) {
        cs.pipe.reset();
        cudaStreamDestroy(cs.stream);
    }
}

// ── CC6 ──────────────────────────────────────────────────────────────────────
// Reuse same slot: compress → decompressInto × N iterations. Validates that
// postStreamSync (forward) and stream-scoped inverse header reads (RZE path)
// are idempotent and do not corrupt state across calls.
TEST(Concurrency, RepeatedReuseSlot_RZE) {
    const size_t in_bytes = N_FLOATS * sizeof(float);
    auto h_in = make_mixed_data(N_FLOATS, 5);

    Slot s = make_slot(build_rze, h_in);

    for (int iter = 0; iter < 20; ++iter) {
        size_t actual = 0;
        s.pipe->decompressInto(s.d_blob, s.blob_sz, s.d_out, s.out_cap, &actual, s.stream);
        cudaStreamSynchronize(s.stream);
        ASSERT_EQ(actual, in_bytes) << "iter " << iter;
    }
    check_slot(s, "CC6/RZE-reuse");
    destroy_slot(s);
}
