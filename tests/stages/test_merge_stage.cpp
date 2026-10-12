/**
 * tests/stages/test_merge_stage.cpp
 *
 * GPU unit + integration tests for MergeStage — N→1 concatenation (forward),
 * 1→N split (inverse).
 *
 *   MG1  MergeStage/ConcatThenSplitRoundTrip   — 3 segments merge + split byte-exact
 *   MG2  MergeStage/EmptySegmentRoundTrip       — a zero-size middle segment round-trips
 *   MG3  MergeStage/HeaderSerialization         — names + sizes survive serialize/deserialize
 *   MG4  MergeStage/PortCounts                  — N→1 forward, 1→N inverse
 *   MG5  MergeStage/IsGraphCompatible           — true in both directions
 *   MG6  MergeStage/PipelineReversal            — LorenzoQuant(1→3) → Merge(3→1) float round-trip
 *   MG7  MergeStage/QuantizerMaterializedTails   — unused outlier capacities stay defined across reuse
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "structural/merge/merge_stage.h"
#include "fzgpumodules.h"

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <random>

using namespace fz;
using namespace fz_test;

// Merge `segs` (forward), then split the blob back (inverse); assert byte-exact.
static void merge_split_round_trip(const std::vector<std::vector<uint8_t>>& segs) {
    CudaStream cs;
    size_t total = 0;
    for (auto& s : segs) total += s.size();
    auto pool = make_test_pool(total + 65536);

    std::vector<std::string> names;
    for (size_t i = 0; i < segs.size(); i++) names.push_back("seg" + std::to_string(i));

    // ── Forward: concat ──
    std::vector<CudaBuffer<uint8_t>> d_segs;
    std::vector<void*>  fwd_in;
    std::vector<size_t> fwd_sz;
    for (auto& s : segs) {
        d_segs.emplace_back(s.empty() ? 1 : s.size());
        if (!s.empty()) d_segs.back().upload(s, cs.stream);
        fwd_in.push_back(d_segs.back().void_ptr());
        fwd_sz.push_back(s.size());
    }
    cudaStreamSynchronize(cs.stream);

    CudaBuffer<uint8_t> d_blob(total ? total : 1);
    std::vector<void*> fwd_out = {d_blob.void_ptr()};

    MergeStage enc;
    enc.setSegmentNames(names);
    enc.execute(cs.stream, pool.get(), fwd_in, fwd_out, fwd_sz);
    cudaStreamSynchronize(cs.stream);
    EXPECT_EQ(enc.getActualOutputSize(0), total);

    // ── Inverse: split (fresh object via serialized header) ──
    uint8_t hdr[FZM_STAGE_CONFIG_SIZE] = {};
    const size_t hsz = enc.serializeHeader(0, hdr, sizeof(hdr));
    ASSERT_GT(hsz, 0u);

    MergeStage dec;
    dec.deserializeHeader(hdr, hsz);
    dec.setInverse(true);
    ASSERT_EQ(dec.getNumOutputs(), segs.size());

    std::vector<CudaBuffer<uint8_t>> d_out;
    std::vector<void*> inv_out;
    for (size_t i = 0; i < segs.size(); i++) {
        d_out.emplace_back(segs[i].empty() ? 1 : segs[i].size());
        inv_out.push_back(d_out.back().void_ptr());
    }
    std::vector<void*>  inv_in = {d_blob.void_ptr()};
    std::vector<size_t> inv_sz = {total};
    dec.execute(cs.stream, pool.get(), inv_in, inv_out, inv_sz);
    cudaStreamSynchronize(cs.stream);

    for (size_t i = 0; i < segs.size(); i++) {
        if (segs[i].empty()) continue;
        std::vector<uint8_t> got(segs[i].size());
        cudaMemcpy(got.data(), d_out[i].get(), segs[i].size(), cudaMemcpyDeviceToHost);
        EXPECT_EQ(got, segs[i]) << "segment " << i << " mismatch";
    }
}

TEST(MergeStage, ConcatThenSplitRoundTrip) {
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> d(0, 255);
    std::vector<std::vector<uint8_t>> segs(3);
    for (size_t i = 0; i < 3; i++) {
        segs[i].resize(1000 + 777 * i);
        for (auto& b : segs[i]) b = (uint8_t)d(rng);
    }
    merge_split_round_trip(segs);
}

TEST(MergeStage, EmptySegmentRoundTrip) {
    std::vector<std::vector<uint8_t>> segs = {
        std::vector<uint8_t>(512, 0xAB),
        std::vector<uint8_t>(),            // empty middle segment (e.g. no outliers)
        std::vector<uint8_t>(300, 0xCD),
    };
    merge_split_round_trip(segs);
}

TEST(MergeStage, HeaderSerialization) {
    MergeStage s;
    s.setSegmentNames({"codes", "anchor", "outliers"});
    // Inject sizes via a forward pass is overkill; serialize zero sizes + names,
    // confirm names + count survive the round-trip.
    uint8_t buf[FZM_STAGE_CONFIG_SIZE] = {};
    const size_t n = s.serializeHeader(0, buf, sizeof(buf));
    ASSERT_GT(n, 0u);
    MergeStage s2;
    s2.deserializeHeader(buf, n);
    s2.setInverse(true);
    ASSERT_EQ(s2.getSegmentNames().size(), 3u);
    EXPECT_EQ(s2.getSegmentNames()[0], "codes");
    EXPECT_EQ(s2.getSegmentNames()[2], "outliers");
}

TEST(MergeStage, PortCounts) {
    MergeStage s;
    s.setSegmentNames({"a", "b", "c", "d"});
    EXPECT_EQ(s.getNumInputs(), 4u);
    EXPECT_EQ(s.getNumOutputs(), 1u);
    s.setInverse(true);
    EXPECT_EQ(s.getNumInputs(), 1u);
    EXPECT_EQ(s.getNumOutputs(), 4u);
}

TEST(MergeStage, IsGraphCompatible) {
    MergeStage s;
    EXPECT_TRUE(s.isGraphCompatible());
    s.setInverse(true);
    EXPECT_TRUE(s.isGraphCompatible());
}

// Integration: validate the inverse-DAG reversal of an N-input-forward stage.
// LorenzoQuant emits 3 ports (codes, outlier_errors, outlier_indices); Merge
// concatenates them into the archive; decompress must split + feed the inverse.
TEST(MergeStage, PipelineReversal) {
    const size_t nx = 256, ny = 256;
    auto h_in = make_smooth_data<float>(nx * ny);
    CudaStream cs;

    Pipeline p(h_in.size() * sizeof(float), MemoryStrategy::PREALLOCATE, 4.0f);
    p.setDims(nx, ny, 1);
    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(1e-3f);
    lq->setErrorBoundMode(ErrorBoundMode::ABS);
    lq->setQuantRadius(32768);
    lq->setOutlierCapacity(0.10f);

    auto* mg = p.addStage<MergeStage>();
    mg->setSegmentNames({"codes", "outlier_errors", "outlier_indices"});
    p.connect(mg, lq, "codes");
    p.connect(mg, lq, "outlier_errors");
    p.connect(mg, lq, "outlier_indices");
    p.finalize();

    auto res = pipeline_round_trip<float>(p, h_in, cs.stream);
    ASSERT_EQ(res.data.size(), h_in.size());
    EXPECT_LE(res.max_error, 1e-3f * 1.01f) << "Lorenzo→Merge round-trip exceeded error bound";
}

TEST(MergeStage, QuantizerMaterializedTails) {
    constexpr size_t count = 4096;
    constexpr float bound = 1e-3f;
    for (auto strategy : {MemoryStrategy::PREALLOCATE, MemoryStrategy::MINIMAL}) {
        SCOPED_TRACE(strategy == MemoryStrategy::PREALLOCATE ? "PREALLOCATE" : "MINIMAL");
        CudaStream stream;
        Pipeline p(count * sizeof(float), strategy, 4.0f);
        p.setSpecializationPolicy(SpecializationPolicy::Off);
        auto* q = p.addStage<QuantizerStage<float, uint16_t>>();
        q->setErrorBound(bound);
        q->setErrorBoundMode(ErrorBoundMode::ABS);
        q->setQuantRadius(512);
        q->setOutlierCapacity(0.10f);
        auto* merge = p.addStage<MergeStage>();
        merge->setSegmentNames({"codes", "outlier_vals", "outlier_idxs"});
        p.connect(merge, q, "codes");
        p.connect(merge, q, "outlier_vals");
        p.connect(merge, q, "outlier_idxs");
        p.finalize();
        const size_t capacity = static_cast<size_t>(std::ceil(count * q->getOutlierCapacity()));
        const size_t values_offset = count * sizeof(uint16_t);
        const size_t indices_offset = values_offset + capacity * sizeof(float);
        const size_t expected_size = indices_offset + capacity * sizeof(uint32_t);

        // A sparse-outlier block followed by an outlier-free block exercises both
        // initial unused capacity and stale valid prefixes from a previous call.
        for (bool outliers : {true, false}) {
            std::vector<float> input(count, 0.125f);
            if (outliers)
                for (size_t i = 0; i < count; i += 97) input[i] = 50.0f;
            CudaBuffer<float> device_input(count);
            device_input.upload(input, stream);
            void* compressed = nullptr;
            size_t compressed_bytes = 0;
            p.compress(device_input.void_ptr(), count * sizeof(float),
                       &compressed, &compressed_bytes, stream);
            ASSERT_EQ(compressed_bytes, expected_size);
            const size_t actual = q->getActualOutlierCount();
            if (outliers) EXPECT_GT(actual, 0u);
            else EXPECT_EQ(actual, 0u);
            ASSERT_LT(actual, capacity);
            std::vector<uint8_t> payload(compressed_bytes);
            FZ_TEST_CUDA(cudaMemcpy(payload.data(), compressed, compressed_bytes,
                                   cudaMemcpyDeviceToHost));
            auto zero = [](uint8_t byte) { return byte == 0; };
            EXPECT_TRUE(std::all_of(payload.begin() + values_offset + actual * sizeof(float),
                                    payload.begin() + indices_offset, zero));
            EXPECT_TRUE(std::all_of(payload.begin() + indices_offset + actual * sizeof(uint32_t),
                                    payload.end(), zero));
            void* reconstructed = nullptr;
            size_t reconstructed_bytes = 0;
            p.decompress(compressed, compressed_bytes, &reconstructed,
                         &reconstructed_bytes, stream);
            ASSERT_EQ(reconstructed_bytes, count * sizeof(float));
            std::vector<float> output(count);
            FZ_TEST_CUDA(cudaMemcpy(output.data(), reconstructed, reconstructed_bytes,
                                   cudaMemcpyDeviceToHost));
            EXPECT_LE(max_abs_error(input, output), bound * 1.01f);
        }
    }
}
