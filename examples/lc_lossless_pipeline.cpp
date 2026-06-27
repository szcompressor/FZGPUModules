/**
 * LC lossless pipeline (cuSZ-Hi back-end) — modular FZGM port.
 *
 * Reproduces cuSZ-Hi's two LC ("Lossless Components", Burtscher et al.) back-end
 * configurations as modular FZGM pipelines:
 *
 *   throughput (tp):  codes  → TCMS1 → BIT1 → RRE1          (Huffman-free)
 *                     outliers ─┐
 *                               └─ Merge → BIT4 → RRE2 → RZE1   (the "BITR" chain)
 *
 *   ratio (cr):       codes  → Huffman ─┐
 *                     outliers ─────────┴─ Merge → RRE4 → TCMS8 → RZE1  (the "RTR" chain)
 *
 * Stage mapping:  TCMS = ZigzagStage (byte-transparent), BIT = BitshuffleStage,
 * RRE = RREStage, RZE = RZEStage(levels=1), Huffman = HuffmanStage.
 *
 * ── Why the Merge stage ──────────────────────────────────────────────────────
 * cuSZ-Hi does NOT compress only the main code stream — it runs an LC chain over
 * the *concatenation* of several producer outputs: in tp mode one chain over
 * [anchor | outliers]; in cr mode one chain over [Huffman(codes) | anchor |
 * outliers].  FZGM's DAG otherwise compresses each port independently and only
 * concatenates leaf outputs into the archive *after* compression — too late to
 * feed a downstream chain.  MergeStage materialises that concatenation *inside*
 * the pipeline (N inputs → 1 contiguous buffer) so a single LC chain can compress
 * it as one stream, and on decompress it splits the reconstructed blob back into
 * the original segments to feed the inverse producers.  Without it there is no
 * way to express "compress the concatenation of these ports as one stream", so
 * the merged-blob layout — and thus a 1-to-1 port of cuSZ-Hi's tp/cr modes —
 * would be impossible.
 *
 * (cuSZ-Hi's default predictor is the spline, which also emits an "anchor" port;
 * GInterpStage exposes it and it would simply be one more Merge input.  This
 * example uses LorenzoQuant for simplicity — its two outlier ports already
 * exercise the merge.)
 *
 * Usage:   ./lc_lossless_pipeline [file.f32 [nx [ny [eb]]]]
 *          (no file → a synthetic smooth field is generated)
 *
 * Build:   cmake --preset release -DBUILD_EXAMPLES=ON && cmake --build build/release
 *          Binary: build/release/bin/examples/lc_lossless_pipeline
 */

#include "fzgpumodules.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace fz;

// ── Load a raw float32 file, or synthesize a smooth field. ───────────────────
static std::vector<float> load_or_make(const std::string& path, size_t n) {
    std::vector<float> v(n);
    if (!path.empty()) {
        std::ifstream f(path, std::ios::binary);
        if (f) {
            f.seekg(0, std::ios::end);
            size_t bytes = static_cast<size_t>(f.tellg());
            f.seekg(0);
            v.resize(bytes / sizeof(float));
            f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(float));
            return v;
        }
        std::fprintf(stderr, "warning: could not open %s, synthesizing\n", path.c_str());
    }
    for (size_t i = 0; i < n; i++)
        v[i] = 100.0f * std::sin(i * 0.01) + 10.0f * std::cos(i * 0.0007);
    return v;
}

static float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// Compress + decompress a built (but not finalized) pipeline; report result.
static void run(const char* label, Pipeline& p,
                const std::vector<float>& host, float eb) {
    const size_t in_bytes = host.size() * sizeof(float);
    p.finalize();

    float* d_in = nullptr;
    cudaMalloc(&d_in, in_bytes);
    cudaMemcpy(d_in, host.data(), in_bytes, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void* d_comp = nullptr;  size_t comp_sz = 0;
    p.compress(d_in, in_bytes, &d_comp, &comp_sz, stream);

    void* d_dec = nullptr;   size_t dec_sz = 0;
    p.decompress(d_comp, comp_sz, &d_dec, &dec_sz, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> recon(dec_sz / sizeof(float));
    cudaMemcpy(recon.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);

    const float err   = max_abs_error(host, recon);
    const double ratio = static_cast<double>(in_bytes) / static_cast<double>(comp_sz);
    const bool   ok    = (recon.size() == host.size()) && (err <= eb * 1.05f);

    std::printf("  %-10s  CR %6.2fx   max|err| %.3e   %s\n",
                label, ratio, err, ok ? "OK" : "*** FAILED ***");

    cudaFree(d_in);
    cudaStreamDestroy(stream);
}

// ── tp: codes→TCMS1→BIT1→RRE1 ; outliers→Merge→BIT4→RRE2→RZE1 ────────────────
static void build_tp(Pipeline& p, size_t nx, size_t ny, float eb) {
    p.setDims(nx, ny, 1);
    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(eb);
    lq->setErrorBoundMode(ErrorBoundMode::ABS);
    lq->setQuantRadius(32768);
    lq->setOutlierCapacity(0.10f);

    // codes chain: TCMS1 → BIT1 → RRE1
    auto* tcms = p.addStage<ZigzagStage<int8_t>>();
    tcms->setByteTransparent(true);                 // LC TCMS on the raw byte stream
    auto* bit  = p.addStage<BitshuffleStage>();
    bit->setElementWidth(1);
    auto* rre1 = p.addStage<RREStage>();
    rre1->setWordSize(1);
    p.connect(tcms, lq, "codes");
    p.connect(bit,  tcms);
    p.connect(rre1, bit);

    // outlier chain (BITR): Merge(outliers) → BIT4 → RRE2 → RZE1
    auto* merge = p.addStage<MergeStage>();
    merge->setSegmentNames({"outlier_errors", "outlier_indices"});
    p.connect(merge, lq, "outlier_errors");
    p.connect(merge, lq, "outlier_indices");
    auto* bit4 = p.addStage<BitshuffleStage>();
    bit4->setElementWidth(4);
    auto* rre2 = p.addStage<RREStage>();
    rre2->setWordSize(2);
    // LC RZE_1 = zero-elimination at 1-byte word granularity, followed by the
    // standard recursive bitmap compression. FZGM's RZEStage is byte-granular
    // (= RZE_1's word size) and levels=4 (default) is exactly that full
    // recursion, so this is a faithful RZE_1.
    auto* rze  = p.addStage<RZEStage>();
    p.connect(bit4, merge);
    p.connect(rre2, bit4);
    p.connect(rze,  rre2);
}

// ── cr: codes→Huffman ; Merge(vle,outliers)→RRE4→TCMS8→RZE1 ──────────────────
static void build_cr(Pipeline& p, size_t nx, size_t ny, float eb) {
    p.setDims(nx, ny, 1);
    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(eb);
    lq->setErrorBoundMode(ErrorBoundMode::ABS);
    lq->setQuantRadius(2048);                        // keep 2*radius within the Huffman codebook
    lq->setOutlierCapacity(0.10f);
    lq->setZigzagCodes(true);                       // Huffman needs contiguous [0, 2r-2] symbols

    auto* huf = p.addStage<HuffmanStage<uint16_t>>();
    huf->setBklen(4096);                             // = 2*radius; covers zigzag codes in [0, 2r-2]
    p.connect(huf, lq, "codes");

    // merged blob: [Huffman(codes) | outlier_errors | outlier_indices]
    auto* merge = p.addStage<MergeStage>();
    merge->setSegmentNames({"vle", "outlier_errors", "outlier_indices"});
    p.connect(merge, huf, "output");
    p.connect(merge, lq,  "outlier_errors");
    p.connect(merge, lq,  "outlier_indices");

    // RTR chain over the blob: RRE4 → TCMS8 → RZE1
    auto* rre4 = p.addStage<RREStage>();
    rre4->setWordSize(4);
    auto* tcms8 = p.addStage<ZigzagStage<int64_t>>();
    tcms8->setByteTransparent(true);
    auto* rze = p.addStage<RZEStage>();   // faithful RZE_1 (byte word size, 4-level recursion)
    p.connect(rre4, merge);
    p.connect(tcms8, rre4);
    p.connect(rze,   tcms8);
}

int main(int argc, char** argv) {
    const std::string file = (argc > 1) ? argv[1] : "";
    const size_t nx = (argc > 2) ? std::stoul(argv[2]) : 512;
    const size_t ny = (argc > 3) ? std::stoul(argv[3]) : 512;
    const float  eb = (argc > 4) ? std::stof(argv[4]) : 1e-2f;

    auto host = load_or_make(file, nx * ny);
    std::printf("LC lossless pipeline — %zu elements (%.1f KB), abs eb = %.1e\n",
                host.size(), host.size() * sizeof(float) / 1024.0, eb);
    std::printf("  %-10s  %-10s  %-16s  %s\n", "mode", "ratio", "accuracy", "status");

    {
        Pipeline p(host.size() * sizeof(float), MemoryStrategy::PREALLOCATE, 4.0f);
        build_tp(p, nx, ny, eb);
        run("tp", p, host, eb);
    }
    {
        Pipeline p(host.size() * sizeof(float), MemoryStrategy::PREALLOCATE, 4.0f);
        build_cr(p, nx, ny, eb);
        run("cr", p, host, eb);
    }
    return 0;
}
