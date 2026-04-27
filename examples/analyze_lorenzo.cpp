/**
 * examples/analyze_lorenzo.cpp
 *
 * Standalone Lorenzo code-distribution analyser.
 *
 * Reads a raw-binary float32 file, runs LorenzoQuantStage directly (no pipeline),
 * and prints statistics + a histogram of the quantisation codes.
 *
 * Usage:
 *   ./analyze_lorenzo <file.bin> [options]
 *
 * Options:
 *   --eb <val>              Absolute error bound  (default: 1e-3)
 *   --eb-mode abs|rel|noa  Error-bound mode      (default: abs)
 *   --radius <n>            Quantisation radius   (default: 32768)
 *   --dims <x> [y [z]]     Spatial dimensions, FAST dim first.
 *                           For row-major NxM data: --dims M N
 *                           (x = columns = fast/innermost, y = rows)
 *   --bins <n>              Histogram half-width in codes (default: 32).
 *                           If ≤5% of codes fall in the window an
 *                           auto-bucketed wide histogram is shown instead.
 *   --no-hist               Skip histogram output
 *
 * Build:  appears automatically in the CMake build as target "analyze_lorenzo"
 */

#include "predictors/lorenzo_quant/lorenzo_quant.h"
#include "mem/mempool.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Minimal RAII wrappers (no GTest dependency)
// ─────────────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(expr) do {                                               \
    cudaError_t _e = (expr);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        std::exit(1);                                                       \
    }                                                                       \
} while (0)

struct DevBuf {
    void*  ptr  = nullptr;
    size_t bytes = 0;

    DevBuf() = default;
    explicit DevBuf(size_t n) : bytes(n) {
        if (n) CUDA_CHECK(cudaMalloc(&ptr, n));
    }
    ~DevBuf() { if (ptr) cudaFree(ptr); }

    DevBuf(DevBuf&& o) noexcept : ptr(o.ptr), bytes(o.bytes) { o.ptr = nullptr; o.bytes = 0; }
    DevBuf& operator=(DevBuf&&) = delete;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

static std::vector<float> read_floats(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    auto sz = f.tellg();
    f.seekg(0);
    if (sz % sizeof(float) != 0)
        throw std::runtime_error("File size is not a multiple of 4 (expected raw float32)");
    std::vector<float> v(sz / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), sz);
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Simple arg parsing
// ─────────────────────────────────────────────────────────────────────────────

struct Args {
    std::string file;
    float       eb         = 1e-3f;
    fz::ErrorBoundMode eb_mode = fz::ErrorBoundMode::ABS;
    int         radius     = 32768;
    size_t      dim_x = 0, dim_y = 1, dim_z = 1;   // 0 → infer at runtime
    int         hist_half  = 32;
    bool        show_hist  = true;
};

static Args parse_args(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "Usage: analyze_lorenzo <file.bin> [--eb <val>] [--eb-mode abs|rel|noa]\n"
            << "                         [--radius <n>] [--dims <x> [y [z]]]\n"
            << "                         [--bins <n>] [--no-hist]\n";
        std::exit(1);
    }
    Args a;
    a.file = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (++i >= argc) { std::cerr << name << " requires a value\n"; std::exit(1); }
            return argv[i];
        };

        if (arg == "--eb") {
            a.eb = std::stof(next("--eb"));
        } else if (arg == "--eb-mode") {
            std::string m = next("--eb-mode");
            if (m == "abs") a.eb_mode = fz::ErrorBoundMode::ABS;
            else if (m == "rel") a.eb_mode = fz::ErrorBoundMode::REL;
            else if (m == "noa") a.eb_mode = fz::ErrorBoundMode::NOA;
            else { std::cerr << "Unknown eb-mode: " << m << "\n"; std::exit(1); }
        } else if (arg == "--radius") {
            a.radius = std::stoi(next("--radius"));
        } else if (arg == "--dims") {
            a.dim_x = std::stoull(next("--dims"));
            if (i + 1 < argc && argv[i+1][0] != '-') {
                a.dim_y = std::stoull(argv[++i]);
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    a.dim_z = std::stoull(argv[++i]);
                }
            }
        } else if (arg == "--bins") {
            a.hist_half = std::stoi(next("--bins"));
        } else if (arg == "--no-hist") {
            a.show_hist = false;
        } else {
            std::cerr << "Unknown option: " << arg << "\n"; std::exit(1);
        }
    }
    return a;
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics helpers
// ─────────────────────────────────────────────────────────────────────────────

static double shannon_entropy(const std::vector<int32_t>& codes, size_t n_total) {
    std::map<int32_t, size_t> freq;
    for (auto c : codes) freq[c]++;
    double H = 0.0;
    for (auto& [v, cnt] : freq) {
        double p = static_cast<double>(cnt) / n_total;
        H -= p * std::log2(p);
    }
    return H;
}

// Print a fixed-window histogram centred at 0.
// Returns the fraction of codes that fell inside the window.
static double print_fixed_histogram(
    const std::map<int32_t, size_t>& freq,
    int32_t                          half_width,
    size_t                           n_total,
    int                              bar_width = 50)
{
    size_t in_range = 0;
    size_t peak_cnt = 0;
    for (int32_t v = -half_width; v <= half_width; ++v) {
        auto it = freq.find(v);
        size_t cnt = it != freq.end() ? it->second : 0;
        in_range += cnt;
        peak_cnt = std::max(peak_cnt, cnt);
    }

    std::cout << "\nCode histogram (centred, range [" << -half_width
              << ", +" << half_width << "])\n";
    std::cout << std::string(60, '-') << "\n";

    for (int32_t v = -half_width; v <= half_width; ++v) {
        auto it = freq.find(v);
        size_t cnt = it != freq.end() ? it->second : 0;
        double pct = 100.0 * cnt / n_total;
        int bar_len = peak_cnt > 0
                      ? static_cast<int>(bar_width * cnt / peak_cnt)
                      : 0;
        std::cout << std::setw(6) << v << " | "
                  << std::string(bar_len, '#')
                  << std::string(bar_width - bar_len, ' ')
                  << " " << std::setw(8) << cnt
                  << "  (" << std::fixed << std::setprecision(2) << pct << "%)\n";
    }

    size_t out_of_range = n_total - in_range;
    std::cout << std::string(60, '-') << "\n";
    std::cout << "  In range  [" << -half_width << ", +" << half_width << "]: "
              << in_range  << "  (" << std::fixed << std::setprecision(2)
              << 100.0*in_range/n_total  << "%)\n";
    std::cout << "  Out of range: "
              << out_of_range << "  (" << std::setprecision(2)
              << 100.0*out_of_range/n_total << "%)\n";

    return static_cast<double>(in_range) / n_total;
}

// Print a bucketed histogram spanning the full code range.
// Each bucket covers `bucket_size` consecutive code values.
static void print_wide_histogram(
    const std::map<int32_t, size_t>& freq,
    int32_t                          code_min,
    int32_t                          code_max,
    size_t                           n_total,
    int                              n_buckets = 60,
    int                              bar_width  = 50)
{
    int64_t span = static_cast<int64_t>(code_max) - code_min + 1;
    int64_t bucket_size = std::max<int64_t>(1, (span + n_buckets - 1) / n_buckets);

    std::vector<size_t> buckets(n_buckets, 0);
    for (auto& [v, cnt] : freq) {
        int idx = static_cast<int>((static_cast<int64_t>(v) - code_min) / bucket_size);
        if (idx >= 0 && idx < n_buckets) buckets[idx] += cnt;
    }

    size_t peak = *std::max_element(buckets.begin(), buckets.end());

    std::cout << "\nCode histogram (bucketed, " << n_buckets << " buckets, "
              << bucket_size << " codes/bucket)\n";
    std::cout << std::string(70, '-') << "\n";

    for (int i = 0; i < n_buckets; ++i) {
        int32_t lo = code_min + static_cast<int32_t>(i * bucket_size);
        int32_t hi = lo + static_cast<int32_t>(bucket_size) - 1;
        double  pct = 100.0 * buckets[i] / n_total;
        int bar_len = peak > 0
                      ? static_cast<int>(bar_width * buckets[i] / peak)
                      : 0;
        std::cout << std::setw(7) << lo << ".." << std::setw(7) << hi << " | "
                  << std::string(bar_len, '#')
                  << std::string(bar_width - bar_len, ' ')
                  << " " << std::setw(8) << buckets[i]
                  << "  (" << std::fixed << std::setprecision(2) << pct << "%)\n";
    }
    std::cout << std::string(70, '-') << "\n";
}

static void print_histogram(
    const std::vector<int32_t>& codes,
    int32_t                     half_width,
    size_t                      n_total,
    int                         bar_width = 50)
{
    std::map<int32_t, size_t> freq;
    for (auto c : codes) freq[c]++;

    auto [cmin_it, cmax_it] = std::minmax_element(codes.begin(), codes.end());

    // Try the requested fixed window first.
    double coverage = print_fixed_histogram(freq, half_width, n_total, bar_width);

    // If fewer than 5 % of codes are visible, also show a wide bucketed view.
    if (coverage < 0.05) {
        std::cout << "\n[Note: only " << std::fixed << std::setprecision(1)
                  << coverage * 100.0 << "% of codes fall in [" << -half_width
                  << ", +" << half_width << "].  Showing full-range view below.]\n";
        print_wide_histogram(freq, *cmin_it, *cmax_it, n_total, 60, bar_width);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    // ── 1. Load data ──────────────────────────────────────────────────────────
    std::cout << "Loading: " << a.file << "\n";
    std::vector<float> h_data = read_floats(a.file);
    size_t n = h_data.size();
    size_t in_bytes = n * sizeof(float);
    std::cout << "  " << n << " elements  (" << in_bytes / (1024.0*1024.0)
              << " MB)\n";

    // ── 2. Configure stage ───────────────────────────────────────────────────
    fz::LorenzoQuantStage<float, uint16_t>::Config cfg;
    cfg.error_bound = a.eb;
    cfg.quant_radius = a.radius;
    cfg.dims = {a.dim_x, a.dim_y, a.dim_z};
    cfg.eb_mode = a.eb_mode;

    auto eb_mode_str = [&] {
        switch (a.eb_mode) {
            case fz::ErrorBoundMode::ABS: return "ABS";
            case fz::ErrorBoundMode::REL: return "REL";
            case fz::ErrorBoundMode::NOA: return "NOA";
        }
        return "?";
    };

    std::cout << "\nLorenzoQuantStage config:\n"
              << "  eb = " << a.eb << "  (" << eb_mode_str() << ")\n"
              << "  radius = " << a.radius << "\n"
              << "  dims = " << cfg.dims[0] << " x " << cfg.dims[1]
              << " x " << cfg.dims[2] << "\n"
              << "  (dim_x=" << cfg.dims[0] << " is the FAST/column axis; "
              << "for row-major NxM use --dims M N)\n";

    fz::LorenzoQuantStage<float, uint16_t> stage(cfg);

    // ── 3. Allocate device memory ────────────────────────────────────────────
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    DevBuf d_in(in_bytes);
    CUDA_CHECK(cudaMemcpyAsync(d_in.ptr, h_data.data(), in_bytes,
                               cudaMemcpyHostToDevice, stream));

    // Estimate output capacities
    auto est = stage.estimateOutputSizes({in_bytes});
    if (est.size() < 4) {
        std::cerr << "Unexpected estimateOutputSizes result\n";
        return 1;
    }

    DevBuf d_codes  (est[0]);
    DevBuf d_errors (est[1]);
    DevBuf d_indices(est[2]);
    DevBuf d_count  (est[3]);

    // ── 4. Memory pool ───────────────────────────────────────────────────────
    fz::MemoryPoolConfig pool_cfg(in_bytes, /*multiplier=*/5.0f);
    fz::MemoryPool pool(pool_cfg);

    // ── 5. Execute ───────────────────────────────────────────────────────────
    std::cout << "\nRunning Lorenzo forward pass... " << std::flush;

    std::vector<void*> inputs  = {d_in.ptr};
    std::vector<void*> outputs = {d_codes.ptr, d_errors.ptr,
                                   d_indices.ptr, d_count.ptr};
    std::vector<size_t> sizes  = {in_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    stage.postStreamSync(stream);   // reads back actual outlier count

    std::cout << "done.\n\n";

    // ── 6. Retrieve actual sizes ──────────────────────────────────────────────
    auto actual = stage.getActualOutputSizesByName();
    size_t codes_bytes   = actual.count("codes") ? actual.at("codes") : est[0];
    size_t outlier_count_bytes = actual.count("outlier_count")
                                  ? actual.at("outlier_count") : est[3];

    size_t n_codes    = codes_bytes / sizeof(uint16_t);
    size_t n_outliers = 0;
    if (outlier_count_bytes >= sizeof(uint32_t)) {
        uint32_t oc = 0;
        CUDA_CHECK(cudaMemcpy(&oc, d_count.ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        n_outliers = oc;
    }

    // ── 7. Download codes ─────────────────────────────────────────────────────
    std::vector<uint16_t> h_codes_raw(n_codes);
    CUDA_CHECK(cudaMemcpy(h_codes_raw.data(), d_codes.ptr,
                          codes_bytes, cudaMemcpyDeviceToHost));

    // Convert raw uint16 codes to signed values via two's-complement reinterpretation.
    // The kernel stores `static_cast<TCode>(static_cast<int>(delta))` — i.e. a signed
    // int16 placed in a uint16 slot.  Do NOT subtract radius here.
    std::vector<int32_t> h_codes(n_codes);
    for (size_t i = 0; i < n_codes; i++) {
        h_codes[i] = static_cast<int32_t>(static_cast<int16_t>(h_codes_raw[i]));
    }

    // ── 8. Compute stats ──────────────────────────────────────────────────────
    size_t n_zero = std::count(h_codes.begin(), h_codes.end(), 0);
    double entropy = shannon_entropy(h_codes, n_codes);

    // Min / max centred code
    auto [cmin_it, cmax_it] = std::minmax_element(h_codes.begin(), h_codes.end());

    double outlier_rate = static_cast<double>(n_outliers) / n;
    double zero_frac    = static_cast<double>(n_zero)    / n_codes;

    // Theoretical minimum bits/element from entropy
    double bits_per_elem = entropy;

    std::cout << "=== Lorenzo Analysis Results ===\n"
              << "  Elements            : " << n << "\n"
              << "  Codes produced      : " << n_codes << "\n"
              << "  Outliers            : " << n_outliers
              << "  (" << std::fixed << std::setprecision(3)
              << outlier_rate * 100.0 << "%)\n"
              << "  Zero-code fraction  : " << std::setprecision(3)
              << zero_frac * 100.0 << "%\n"
              << "  Code range (int16)  : [" << *cmin_it << ", " << *cmax_it << "]\n"
              << "  Outlier threshold   : |delta| >= " << a.radius
              << " quant units  (= " << a.radius * 2.0f * a.eb << " abs units)\n"
              << "  Shannon entropy     : " << std::setprecision(4)
              << entropy << " bits/symbol\n"
              << "  (minimum " << std::setprecision(2)
              << bits_per_elem * n / (1024.0*1024.0*8.0)
              << " MB if entropy-coded)\n";

    // ── 9. Histogram ──────────────────────────────────────────────────────────
    if (a.show_hist) {
        print_histogram(h_codes, a.hist_half, n_codes);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
