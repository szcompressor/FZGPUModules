/**
 * examples/ownership_example.cpp
 *
 * Device-buffer ownership, made explicit by type.
 *
 * The pipeline's explicit-ownership API states ownership in the *return type*, so
 * the compiler tells you whether you own a buffer — you never have to remember a
 * flag or a comment:
 *
 *   fz::BorrowedDeviceBuffer   pool-owned. You may read it; you must NOT free it.
 *                              Invalidated by the next call that reuses the slot
 *                              (compress/decompress/reset) or Pipeline destruction.
 *
 *   fz::OwnedDeviceBuffer      caller-owned, move-only. Frees itself on scope exit
 *                              through the backend the library was built for (never
 *                              a hard-coded cudaFree). release() hands you the raw
 *                              pointer if you want to manage it yourself.
 *
 *   fz::DeviceSpan / ConstDeviceSpan   a non-owning {ptr, bytes} view you pass in.
 *
 * The calls:
 *   compress(ConstDeviceSpan) -> BorrowedDeviceBuffer         (into the pool)
 *   compressInto(ConstDeviceSpan, DeviceSpan) -> size_t       (into your buffer)
 *   decompressBorrowed(ConstDeviceSpan) -> BorrowedDeviceBuffer
 *   decompressOwned(ConstDeviceSpan)    -> OwnedDeviceBuffer
 *   decompressInto(ConstDeviceSpan, DeviceSpan) -> size_t     (into your buffer)
 *
 * decompressBorrowed()/decompressOwned() ignore setPoolManagedDecompOutput() — the
 * call site decides, not pipeline state. (The older pointer overloads + that flag
 * still work and are documented in docs/api_reference.md, but new code should prefer
 * these.)
 *
 * No external data files required — uses synthetic float data.
 *
 * Build:
 *   cmake --preset release -DBUILD_EXAMPLES=ON && cmake --build build/release -j$(nproc)
 *   Binary: build/release/bin/examples/ownership_example
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace fz;

static constexpr size_t N           = 1 << 18;  // 256 K floats = 1 MB
static constexpr float  ERROR_BOUND = 1e-3f;

static std::vector<float> make_data(size_t n) {
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(n);
        h[i] = std::sin(2.0f * 3.14159265f * t)
             + 0.5f * std::cos(6.0f * 3.14159265f * t);
    }
    return h;
}

static void build_pipeline(Pipeline& p) {
    auto* lorenzo = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lorenzo->setErrorBound(ERROR_BOUND);
    lorenzo->setErrorBoundMode(ErrorBoundMode::PREL);
    auto* rle = p.addStage<RLEStage<uint16_t>>();
    p.connect(rle, lorenzo, "codes");
    p.finalize();
}

// Copies d_rec back to host and prints the max absolute error.
static void print_error(const std::vector<float>& h_orig, const void* d_rec, size_t bytes) {
    std::vector<float> h_rec(bytes / sizeof(float));
    cudaMemcpy(h_rec.data(), d_rec, bytes, cudaMemcpyDeviceToHost);
    float max_err = 0.0f;
    for (size_t i = 0; i < h_rec.size(); ++i)
        max_err = std::max(max_err, std::abs(h_rec[i] - h_orig[i]));
    std::printf("  max_abs_error = %.6f\n", max_err);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    const std::vector<float> h_data = make_data(N);
    const size_t data_bytes = N * sizeof(float);

    float* d_input = nullptr;
    cudaMalloc(&d_input, data_bytes);
    cudaMemcpy(d_input, h_data.data(), data_bytes, cudaMemcpyHostToDevice);
    const ConstDeviceSpan input{d_input, data_bytes};

    // ── Section 1: Borrowed (pool-owned) compress + decompress ────────────────
    //
    // compress() returns a BorrowedDeviceBuffer: the pool owns it, so you never
    // free it. It stays valid only until the next call that reuses its slot
    // (another compress(), a decompress(), reset(), or Pipeline destruction), so
    // consume it before the next such call.
    std::printf("── Section 1: borrowed (pool-owned) buffers ──────────────────────────\n");
    {
        Pipeline p(data_bytes);
        build_pipeline(p);

        BorrowedDeviceBuffer comp1 = p.compress(input);
        cudaDeviceSynchronize();
        std::printf("  compress #1: %zu bytes at %p\n", comp1.bytes(), comp1.data());
        // Use comp1 here (write to file, send over network). Do NOT free it.

        // Compress again — comp1's storage may now be reused; treat it as stale.
        BorrowedDeviceBuffer comp2 = p.compress(input);
        cudaDeviceSynchronize();
        std::printf("  compress #2: %zu bytes at %p\n", comp2.bytes(), comp2.data());

        // Decompress the live buffer to verify. decompressBorrowed() also borrows
        // from the pool — do NOT free the result.
        BorrowedDeviceBuffer dec = p.decompressBorrowed(comp2.cspan());
        cudaDeviceSynchronize();
        print_error(h_data, dec.data(), dec.bytes());
    }

    // ── Section 2: compressInto a caller-provided buffer ──────────────────────
    //
    // When the compressed bytes must outlive a later compress()/reset(), write
    // them into your own buffer. compressInto() returns the byte count written
    // and throws if the buffer is too small. getMaxCompressedSize() is a safe cap.
    std::printf("\n── Section 2: compressInto a caller buffer ────────────────────────────\n");
    {
        Pipeline p(data_bytes);
        build_pipeline(p);

        const size_t max_comp = p.getMaxCompressedSize(data_bytes);
        std::printf("  getMaxCompressedSize: %zu bytes\n", max_comp);

        void* d_comp_buf = nullptr;
        cudaMalloc(&d_comp_buf, max_comp);   // caller owns this for its whole lifetime

        const size_t actual = p.compressInto(input, DeviceSpan{d_comp_buf, max_comp});
        cudaDeviceSynchronize();
        std::printf("  actual compressed:    %zu bytes (%.2fx ratio)\n",
                    actual, static_cast<double>(data_bytes) / actual);

        // d_comp_buf survives any number of later compress()/reset() calls on p.
        void* d_comp_buf2 = nullptr;
        cudaMalloc(&d_comp_buf2, max_comp);
        const size_t actual2 = p.compressInto(input, DeviceSpan{d_comp_buf2, max_comp});
        cudaDeviceSynchronize();
        std::printf("  second compress:      %zu bytes  (first buffer still valid)\n", actual2);

        // Decompress from the first buffer — still intact.
        OwnedDeviceBuffer dec = p.decompressOwned(ConstDeviceSpan{d_comp_buf, actual});
        cudaDeviceSynchronize();
        std::printf("  decompress from first buffer:\n");
        print_error(h_data, dec.data(), dec.bytes());

        cudaFree(d_comp_buf);    // caller-provided buffers are yours to free
        cudaFree(d_comp_buf2);
    }

    // ── Section 3: Borrowed decompress + reset() lifetime ─────────────────────
    //
    // A BorrowedDeviceBuffer from decompressBorrowed() survives reset() (reset()
    // frees the compress output, not the decompress output) but is invalidated by
    // the next decompress(). This is the pool-owned decompress path with the
    // ownership stated at the call site rather than through a pipeline flag.
    std::printf("\n── Section 3: borrowed decompress + reset() lifetime ─────────────────\n");
    {
        Pipeline p(data_bytes);
        build_pipeline(p);

        BorrowedDeviceBuffer comp = p.compress(input);
        cudaDeviceSynchronize();

        BorrowedDeviceBuffer dec1 = p.decompressBorrowed(comp.cspan());
        cudaDeviceSynchronize();
        std::printf("  decompress #1: %zu bytes at %p\n", dec1.bytes(), dec1.data());

        // reset() frees the compress output; dec1 is still valid.
        p.reset();
        std::printf("  after reset(): compress output freed, dec1 still valid\n");
        print_error(h_data, dec1.data(), dec1.bytes());

        // A second decompress reuses dec1's slot — treat dec1 as stale afterward.
        BorrowedDeviceBuffer comp2 = p.compress(input);
        cudaDeviceSynchronize();
        BorrowedDeviceBuffer dec2 = p.decompressBorrowed(comp2.cspan());
        cudaDeviceSynchronize();
        std::printf("  decompress #2: %zu bytes at %p  (dec1 now stale)\n",
                    dec2.bytes(), dec2.data());
    }

    // ── Section 4: Owned decompress output ────────────────────────────────────
    //
    // decompressOwned() hands back an OwnedDeviceBuffer: a fresh allocation you
    // own. It frees itself when it goes out of scope (through the correct backend),
    // so there is no cudaFree to remember and no double-free to risk. Use it when
    // the decompressed data must outlive the pipeline, or when you want a plain
    // owning handle.
    std::printf("\n── Section 4: owned decompress output ────────────────────────────────\n");
    {
        Pipeline p(data_bytes);
        build_pipeline(p);

        BorrowedDeviceBuffer comp = p.compress(input);
        cudaDeviceSynchronize();

        {
            OwnedDeviceBuffer dec = p.decompressOwned(comp.cspan());
            cudaDeviceSynchronize();
            std::printf("  decompress: %zu bytes at %p\n", dec.bytes(), dec.data());
            print_error(h_data, dec.data(), dec.bytes());
        }   // dec frees itself here — no cudaFree, no leak, no double-free
        std::printf("  owned buffer released on scope exit\n");

        // Need the raw pointer instead? release() transfers ownership to you.
        OwnedDeviceBuffer dec2 = p.decompressOwned(comp.cspan());
        cudaDeviceSynchronize();
        void* raw = dec2.release();   // dec2 no longer owns it
        cudaFree(raw);                // now it is yours to free
        std::printf("  release() + cudaFree succeeded\n");
    }

    cudaFree(d_input);   // d_input is caller-owned — must cudaFree
    std::printf("\nDone.\n");
    return 0;
}
