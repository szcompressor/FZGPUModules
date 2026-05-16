/**
 * examples/compare_lorenzo_modes.cpp
 *
 * Runs LorenzoQuantStage<float, uint16_t> with the two supported code-storage
 * modes and writes the resulting uint16 arrays to disk for histogram comparison.
 *
 * Mode 1 — raw delta  (zigzag=false)
 *   code = static_cast<uint16_t>(static_cast<int>(delta))
 *   Positive deltas → [0, radius-1]; negative deltas → [65537-radius, 65535].
 *   The two code lobes require bklen=65536 for Huffman.
 *
 * Mode 2 — zigzag     (zigzag=true)
 *   code = zigzag_encode(delta)  — bijective map to [0, 2*radius-2]
 *   Codes are contiguous; bklen=2*radius is sufficient for Huffman.
 *
 * Both passes use identical settings; only zigzag_codes differs.  Shannon
 * entropy is the same for both (the map is bijective), but the bklen
 * requirement (and thus PHF memory cost) differs by 64×.
 *
 * Usage:
 *   ./compare_lorenzo_modes <file.f32> --dims <x> <y>  [options]
 *
 *   --eb <val>           Error bound (default: 1e-4)
 *   --radius <n>         Quantization radius (default: 512)
 *   --out-rawdelta <p>   Output path for raw-delta codes (default: codes_rawdelta.u16)
 *   --out-zigzag <p>     Output path for zigzag codes   (default: codes_zigzag.u16)
 *
 * Example:
 *   ./compare_lorenzo_modes data/CLDHGH.f32 --dims 3600 1800 --eb 1e-4
 *   python3 examples/plot_lorenzo_codes.py codes_rawdelta.u16 codes_zigzag.u16
 *
 * Build:
 *   cmake --preset release -DBUILD_EXAMPLES=ON
 *   cmake --build build/release -j$(nproc)
 *   Binary: build/release/bin/examples/compare_lorenzo_modes
 */

#include "fused/lorenzo_quant/lorenzo_quant.h"
#include "mem/mempool.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
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
};

// ─────────────────────────────────────────────────────────────────────────────
// I/O
// ─────────────────────────────────────────────────────────────────────────────

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

static void write_u16(const std::string& path, const std::vector<uint16_t>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);
    f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(uint16_t));
    std::cout << "  Saved " << v.size() << " codes → " << path << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Arg parsing
// ─────────────────────────────────────────────────────────────────────────────

struct Args {
    std::string file;
    float       eb           = 1e-4f;
    int         radius       = 512;
    size_t      dim_x = 0, dim_y = 1, dim_z = 1;
    std::string out_rawdelta = "codes_rawdelta.u16";
    std::string out_zigzag   = "codes_zigzag.u16";
};

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <file.f32> --dims <x> <y>  [options]\n"
        << "\n"
        << "  --eb <val>           Error bound (default: 1e-4)\n"
        << "  --radius <n>         Quantization radius (default: 512)\n"
        << "  --out-rawdelta <p>   Output for raw-delta codes (default: codes_rawdelta.u16)\n"
        << "  --out-zigzag <p>     Output for zigzag codes   (default: codes_zigzag.u16)\n"
        << "\n"
        << "Example:\n"
        << "  " << prog << " data/CLDHGH.f32 --dims 3600 1800 --eb 1e-4\n"
        << "  python3 examples/plot_lorenzo_codes.py codes_rawdelta.u16 codes_zigzag.u16\n";
    std::exit(1);
}

static Args parse_args(int argc, char** argv) {
    if (argc < 2) usage(argv[0]);
    Args a;
    a.file = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (++i >= argc) { std::cerr << name << " requires a value\n"; std::exit(1); }
            return argv[i];
        };
        if      (arg == "--eb")           a.eb     = std::stof(next("--eb"));
        else if (arg == "--radius")       a.radius = std::stoi(next("--radius"));
        else if (arg == "--out-rawdelta") a.out_rawdelta = next("--out-rawdelta");
        else if (arg == "--out-zigzag")   a.out_zigzag   = next("--out-zigzag");
        else if (arg == "--dims") {
            a.dim_x = std::stoull(next("--dims"));
            if (i + 1 < argc && argv[i+1][0] != '-') a.dim_y = std::stoull(argv[++i]);
            if (i + 1 < argc && argv[i+1][0] != '-') a.dim_z = std::stoull(argv[++i]);
        } else { std::cerr << "Unknown option: " << arg << "\n"; usage(argv[0]); }
    }
    return a;
}

// ─────────────────────────────────────────────────────────────────────────────
// Run one Lorenzo forward pass — returns host uint16 code array
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<uint16_t> run_pass(
    const std::vector<float>& h_data,
    const Args&               a,
    bool                      zigzag,
    cudaStream_t              stream,
    size_t&                   out_n_outliers)
{
    const size_t in_bytes = h_data.size() * sizeof(float);

    fz::LorenzoQuantStage<float, uint16_t>::Config cfg;
    cfg.error_bound      = a.eb;
    cfg.quant_radius     = a.radius;
    cfg.outlier_capacity = 0.2f;
    cfg.dims             = {a.dim_x, a.dim_y, a.dim_z};
    cfg.eb_mode          = fz::ErrorBoundMode::ABS;
    cfg.zigzag_codes     = zigzag;

    fz::LorenzoQuantStage<float, uint16_t> stage(cfg);

    DevBuf d_in(in_bytes);
    CUDA_CHECK(cudaMemcpyAsync(d_in.ptr, h_data.data(), in_bytes,
                               cudaMemcpyHostToDevice, stream));

    auto est = stage.estimateOutputSizes({in_bytes});
    DevBuf d_codes  (est[0]);
    DevBuf d_errors (est[1]);
    DevBuf d_indices(est[2]);
    DevBuf d_count  (est[3]);

    fz::MemoryPoolConfig pool_cfg(in_bytes, 5.0f);
    fz::MemoryPool pool(pool_cfg);

    std::vector<void*>  inputs  = {d_in.ptr};
    std::vector<void*>  outputs = {d_codes.ptr, d_errors.ptr, d_indices.ptr, d_count.ptr};
    std::vector<size_t> sizes   = {in_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    stage.postStreamSync(stream);

    auto actual       = stage.getActualOutputSizesByName();
    size_t codes_bytes = actual.count("codes") ? actual.at("codes") : est[0];

    out_n_outliers = 0;
    if (actual.count("outlier_count") && actual.at("outlier_count") >= sizeof(uint32_t)) {
        uint32_t oc = 0;
        CUDA_CHECK(cudaMemcpy(&oc, d_count.ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        out_n_outliers = oc;
    }

    size_t n_codes = codes_bytes / sizeof(uint16_t);
    std::vector<uint16_t> h_codes(n_codes);
    CUDA_CHECK(cudaMemcpy(h_codes.data(), d_codes.ptr, codes_bytes, cudaMemcpyDeviceToHost));
    return h_codes;
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats
// ─────────────────────────────────────────────────────────────────────────────

static void print_stats(
    const char*                  label,
    const std::vector<uint16_t>& codes,
    bool                         zigzag,
    int                          radius,
    size_t                       n_outliers,
    size_t                       n_total)
{
    // Build frequency map in the semantic (signed delta) domain for entropy.
    // Both modes encode the same deltas so entropy is identical.
    std::map<int32_t, size_t> freq;
    for (auto c : codes) {
        int32_t key = zigzag
            ? static_cast<int32_t>(c)
            : static_cast<int32_t>(static_cast<int16_t>(c));
        freq[key]++;
    }

    double H = 0.0;
    for (auto& [v, cnt] : freq) {
        double p = static_cast<double>(cnt) / codes.size();
        H -= p * std::log2(p);
    }

    auto [cmin_u, cmax_u] = std::minmax_element(codes.begin(), codes.end());

    // Signed range (only meaningful for raw-delta mode)
    int16_t smin = INT16_MAX, smax = INT16_MIN;
    if (!zigzag) {
        for (auto c : codes) {
            int16_t s = static_cast<int16_t>(c);
            if (s < smin) smin = s;
            if (s > smax) smax = s;
        }
    }

    std::cout << "\n--- " << label << " ---\n"
              << "  Elements          : " << n_total << "\n"
              << "  Codes array       : " << codes.size() << "\n"
              << "  Outliers          : " << n_outliers << "  ("
              << std::fixed << std::setprecision(3)
              << 100.0 * n_outliers / n_total << "%)\n"
              << "  uint16 range      : [" << *cmin_u << ", " << *cmax_u << "]\n";

    if (!zigzag) {
        std::cout
            << "  Signed int16 range: [" << smin << ", " << smax << "]\n"
            << "  Codes form two lobes in uint16 space:\n"
            << "    positive deltas → [0, " << radius - 1 << "]\n"
            << "    negative deltas → [" << 65537 - radius << ", 65535]\n"
            << "  Huffman bklen req : 65536\n";
    } else {
        std::cout
            << "  Codes are contiguous in [0, " << 2 * radius - 2 << "]\n"
            << "    even code k   → delta = +k/2\n"
            << "    odd  code k   → delta = -(k+1)/2\n"
            << "  Huffman bklen req : " << 2 * radius << "\n";
    }

    std::cout
        << "  Shannon entropy   : " << std::setprecision(4) << H << " bits/sym\n"
        << "  Est. coded size   : " << std::setprecision(2)
        << H * codes.size() / (8.0 * 1024.0 * 1024.0) << " MB\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    std::cout << "Loading: " << a.file << "\n";
    std::vector<float> h_data = read_floats(a.file);
    const size_t n = h_data.size();
    std::cout << "  " << n << " floats  ("
              << n * sizeof(float) / (1024.0 * 1024.0) << " MB)\n"
              << "  dims = " << a.dim_x << " x " << a.dim_y << " x " << a.dim_z << "\n"
              << "  eb = " << a.eb << "  radius = " << a.radius << "\n";

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Pass 1: raw delta ─────────────────────────────────────────────────────
    std::cout << "\nPass 1: raw delta (zigzag=false)...\n";
    size_t n_outliers_1 = 0;
    auto codes_rawdelta = run_pass(h_data, a, false, stream, n_outliers_1);
    print_stats("Mode 1: raw delta (zigzag=false)",
                codes_rawdelta, false, a.radius, n_outliers_1, n);

    // ── Pass 2: zigzag ────────────────────────────────────────────────────────
    std::cout << "\nPass 2: zigzag (zigzag=true)...\n";
    size_t n_outliers_2 = 0;
    auto codes_zigzag = run_pass(h_data, a, true, stream, n_outliers_2);
    print_stats("Mode 2: zigzag    (zigzag=true)",
                codes_zigzag, true, a.radius, n_outliers_2, n);

    // ── Write binary outputs ──────────────────────────────────────────────────
    std::cout << "\nWriting binary code arrays:\n";
    write_u16(a.out_rawdelta, codes_rawdelta);
    write_u16(a.out_zigzag,   codes_zigzag);

    std::cout
        << "\nNote: Shannon entropy is identical for both modes (bijective map).\n"
        << "      The difference is bklen: raw-delta requires 65536, zigzag requires "
        << 2 * a.radius << ".\n"
        << "\nVisualize with:\n"
        << "  python3 examples/plot_lorenzo_codes.py "
        << a.out_rawdelta << " " << a.out_zigzag << " --radius " << a.radius << "\n";

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
