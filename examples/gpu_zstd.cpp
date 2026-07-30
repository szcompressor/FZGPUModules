/**
 * examples/gpu_zstd.cpp
 *
 * GPU Zstandard: LorenzoQuant -> GPULZ (split) -> {Huffman, ANS x3} -> Merge
 *
 * Reproduces Zstandard's block structure on the GPU. Zstd splits a compressed
 * block into a *literals* section (Huffman coded) and a *sequences* section
 * (FSE coded), because the two carry very different symbol distributions and
 * interleaving them raises the entropy any single coder sees. `GPULZStage` in
 * split mode emits that separation directly as four ports:
 *
 *     literals -> HuffmanStage<uint16_t>   (Zstd: Huffman over literals)
 *     lengths  -> ANSStage                 (Zstd: FSE over Match_Length codes)
 *     offsets  -> ANSStage                 (Zstd: FSE over offset codes)
 *     meta     -> ANSStage                 (header + per-chunk flag bitmaps)
 *
 * then MergeStage concatenates the four coded sub-streams into one archive.
 *
 * Two deliberate departures from the CPU format:
 *
 *   - Zstd interleaves the three sequence streams into a single bitstream with
 *     three FSE states, because on a CPU that keeps all three states in
 *     registers through one sequential decode loop. On a GPU the tradeoff
 *     inverts: same-level DAG stages run concurrently on separate streams, so
 *     keeping the streams separate lets their coders overlap.
 *   - Zstd encodes lengths/offsets as a small *code* plus raw *extra bits* to
 *     keep the FSE alphabet tiny while representing values past 65536. Our
 *     length and offset fields are one byte each, so the alphabet is already
 *     <=256 and the extra-bits split buys nothing. It would become necessary
 *     if the match window were widened past 255.
 *
 * The literals port keeps the data's natural uint16 quant-code alphabet, which
 * is why it goes to HuffmanStage<uint16_t> rather than a byte coder -- coding
 * those codes as bytes throws away the correlation between a code's high and
 * low byte and measured ~20% worse on the literals stream alone.
 *
 * Usage:
 *   ./build/bin/examples/gpu_zstd <file.f32> <n_floats> <nx> <ny> <nz> [radius]
 *   ./build/bin/examples/gpu_zstd            # synthetic data, no file needed
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace fz;

namespace {

struct Result {
    const char* name;
    size_t      comp_bytes;
    double      cr;
    float       comp_ms;
    float       decomp_ms;
    float       max_err;
};

// Compress + decompress one pipeline, timing both directions.
Result run(const char* name, Pipeline& p, const float* d_input,
           size_t in_bytes, const std::vector<float>& h_ref, cudaStream_t stream)
{
    Result r{name, 0, 0.0, 0.0f, 0.0f, 0.0f};

    void*  d_comp = nullptr; size_t comp_sz = 0;
    void*  d_dec  = nullptr; size_t dec_sz  = 0;
    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);

    // warm-up pass (first launch pays module load and pool growth)
    p.compress(d_input, in_bytes, &d_comp, &comp_sz, stream);
    p.decompress(d_comp, comp_sz, &d_dec, &dec_sz, stream);
    cudaStreamSynchronize(stream);

    cudaEventRecord(e0, stream);
    p.compress(d_input, in_bytes, &d_comp, &comp_sz, stream);
    cudaEventRecord(e1, stream);
    cudaStreamSynchronize(stream);

    cudaEventRecord(e2, stream);
    p.decompress(d_comp, comp_sz, &d_dec, &dec_sz, stream);
    cudaEventRecord(e3, stream);
    cudaStreamSynchronize(stream);

    cudaEventElapsedTime(&r.comp_ms,   e0, e1);
    cudaEventElapsedTime(&r.decomp_ms, e2, e3);

    if (const char* pv = std::getenv("GPU_ZSTD_PROFILE"); pv && *pv == '1') {
        p.enableProfiling(true);
        p.compress(d_input, in_bytes, &d_comp, &comp_sz, stream);
        cudaStreamSynchronize(stream);
        std::printf("\n--- per-stage COMPRESS: %s ---\n", name);
        p.getLastPerfResult().print(std::cout);
        p.decompress(d_comp, comp_sz, &d_dec, &dec_sz, stream);
        cudaStreamSynchronize(stream);
        std::printf("--- per-stage DECOMPRESS: %s ---\n", name);
        p.getLastPerfResult().print(std::cout);
        p.enableProfiling(false);
    }

    r.comp_bytes = comp_sz;
    r.cr = double(in_bytes) / double(comp_sz);

    std::vector<float> h_out(h_ref.size(), 0.0f);
    const size_t copy_bytes = std::min(dec_sz, h_ref.size() * sizeof(float));
    cudaMemcpy(h_out.data(), d_dec, copy_bytes, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < h_ref.size(); i++)
        r.max_err = std::max(r.max_err, std::fabs(h_out[i] - h_ref[i]));

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(e2); cudaEventDestroy(e3);
    return r;
}

void configureLorenzo(LorenzoQuantStage<float, uint16_t>* lq,
                      int nx, int ny, int nz, float eb, int radius) {
    lq->setErrorBound(eb);
    lq->setErrorBoundMode(ErrorBoundMode::REL);
    lq->setQuantRadius(radius);
    lq->setZigzagCodes(true);
    (void)nx; (void)ny; (void)nz;
}

} // namespace

int main(int argc, char** argv) {
    std::string path;
    size_t n_floats = 1u << 22;
    int nx = (int)n_floats, ny = 1, nz = 1;
    int radius = 2048;   // see the bklen note on the Huffman stage below
    const float eb = 1e-3f;

    if (argc > 1) {
        path     = argv[1];
        n_floats = (argc > 2) ? std::stoul(argv[2]) : n_floats;
        nx = (argc > 3) ? std::stoi(argv[3]) : (int)n_floats;
        ny = (argc > 4) ? std::stoi(argv[4]) : 1;
        nz = (argc > 5) ? std::stoi(argv[5]) : 1;
        radius = (argc > 6) ? std::stoi(argv[6]) : radius;
    }

    std::vector<float> h_in(n_floats);
    if (!path.empty()) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 1; }
        ifs.read(reinterpret_cast<char*>(h_in.data()), n_floats * sizeof(float));
    } else {
        for (size_t i = 0; i < n_floats; i++) {
            const float t = float(i) / float(n_floats);
            h_in[i] = std::sin(50.0f * 3.14159265f * t) + 0.25f * std::cos(311.0f * t);
        }
    }
    const size_t in_bytes = n_floats * sizeof(float);

    cudaStream_t stream; cudaStreamCreate(&stream);
    float* d_in = nullptr;
    cudaMalloc(&d_in, in_bytes);
    cudaMemcpy(d_in, h_in.data(), in_bytes, cudaMemcpyHostToDevice);

    std::printf("input: %s  %zu floats (%.1f MB)  dims %dx%dx%d  eb=%g rel  radius=%d\n\n",
                path.empty() ? "<synthetic>" : path.c_str(), n_floats,
                in_bytes / 1048576.0, nx, ny, nz, eb, radius);

    std::vector<Result> results;

    // ── baseline A: the library's existing lossless back-end ──────────────
    {
        Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
        p.setDims(nx, ny, nz);
        auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
        configureLorenzo(lq, nx, ny, nz, eb, radius);
        auto* bs = p.addStage<BitshuffleStage>();
        bs->setElementWidth(2);
        auto* rze = p.addStage<RZEStage>();
        p.connect(bs, lq, "codes");
        p.connect(rze, bs);
        p.finalize();
        results.push_back(run("lorenzo->bitshuffle->rze", p, d_in, in_bytes, h_in, stream));
    }

    // ── baseline B: single-stream GPULZ, one byte coder over everything ───
    {
        Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
        p.setDims(nx, ny, nz);
        auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
        configureLorenzo(lq, nx, ny, nz, eb, radius);
        auto* lz = p.addStage<GPULZStage>();
        lz->setChunkSize(2048);
        lz->setWordSize(4);
        auto* ans = p.addStage<ANSStage>();
        p.connect(lz, lq, "codes");
        p.connect(ans, lz);
        p.finalize();
        results.push_back(run("lorenzo->gpulz->ans (single)", p, d_in, in_bytes, h_in, stream));
    }

    // ── GPU Zstd: split ports, one coder each, merged ─────────────────────
    {
        Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
        p.setDims(nx, ny, nz);
        auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
        configureLorenzo(lq, nx, ny, nz, eb, radius);

        auto* lz = p.addStage<GPULZStage>();
        lz->setChunkSize(2048);
        lz->setWordSize(4);
        lz->setSplitMode(true);
        p.connect(lz, lq, "codes");

        // Literals keep the uint16 quant-code alphabet -> symbol-width Huffman.
        // Zigzag codes span [0, 2*radius-2], so 2*radius-1 entries suffice.
        //
        // This is what bounds the usable quantization radius in this pipeline:
        // HuffmanStage histograms into *shared* memory, so a codebook of
        // 2*radius entries has to fit there (radius 2048 -> 4096 entries ->
        // 16 KB, fine; radius 32768 -> 64 K entries -> 256 KB, which faults).
        // HuffmanStage also narrows bklen to uint16_t internally, so 65536
        // wraps to 0 and every symbol then reads as out-of-range. If a large
        // radius is required, code the literals with ANSStage instead and give
        // up the uint16-alphabet advantage.
        // 2*radius covers zigzag codes [0, 2*radius-2] with one spare slot.
        // Must stay <= 65535 (HuffmanStage narrows bklen to uint16_t, so 65536
        // wraps to 0 and every symbol reads as out-of-range), which caps the
        // radius at 32767 for this pipeline.
        const uint32_t bklen = std::min<uint32_t>(65534u, uint32_t(2 * radius));
        if (2 * radius - 1 > 8192)
            std::printf("warning: radius %d gives a %u-entry Huffman codebook; "
                        "this may exceed the shared-memory histogram\n", radius, bklen);
        auto* huf = p.addStage<HuffmanStage<uint16_t>>();
        huf->setBklen(bklen);
        if (const char* fm = std::getenv("GPU_ZSTD_HUF_FINE"); fm && *fm == '1')
            huf->setEncodeMode(HuffmanEncodeMode::Fine);
        p.connect(huf, lz, "literals");

        auto* ans_len  = p.addStage<ANSStage>();
        auto* ans_off  = p.addStage<ANSStage>();
        auto* ans_meta = p.addStage<ANSStage>();
        p.connect(ans_len,  lz, "lengths");
        p.connect(ans_off,  lz, "offsets");
        p.connect(ans_meta, lz, "meta");

        // No MergeStage here on purpose. Merge exists for the case where a
        // *downstream* stage must see the concatenation of several ports; here
        // nothing consumes it, and the four coded streams are simply leaf
        // outputs that Pipeline assembles into the archive after all stages
        // run. Merging them early would also hand each coder's inverse an
        // interior pointer at an arbitrary byte offset, and both the Huffman
        // and ANS decoders read their bitstreams as 32-bit words -- Huffman's
        // decode kernel faults on the misaligned read.
        p.finalize();
        results.push_back(run("GPU-Zstd (split, 4 coders)", p, d_in, in_bytes, h_in, stream));
    }

    // ── report ────────────────────────────────────────────────────────────
    std::printf("%-30s %12s %9s %10s %10s %10s\n",
                "pipeline", "bytes", "CR", "comp GB/s", "dec GB/s", "max err");
    for (const auto& r : results) {
        const double cgb = (in_bytes / 1e9) / (r.comp_ms   / 1e3);
        const double dgb = (in_bytes / 1e9) / (r.decomp_ms / 1e3);
        std::printf("%-30s %12zu %8.2fx %10.1f %10.1f %10.2e\n",
                    r.name, r.comp_bytes, r.cr, cgb, dgb, r.max_err);
    }
    if (results.size() == 3) {
        std::printf("\nGPU-Zstd vs bitshuffle->rze : %+.1f%% CR\n",
                    100.0 * (results[2].cr / results[0].cr - 1.0));
        std::printf("GPU-Zstd vs single-stream   : %+.1f%% CR\n",
                    100.0 * (results[2].cr / results[1].cr - 1.0));
    }

    cudaFree(d_in);
    cudaStreamDestroy(stream);
    return 0;
}
