/**
 * examples/eb_mode_analysis.cpp
 *
 * Measures what each error-bound mode actually guarantees, and what it costs.
 *
 * The motivating question: `LorenzoQuantStage` / `GInterpStage` cannot honour a
 * per-element relative bound — they quantize prediction residuals against one
 * global tolerance. Their approximation is `PREL`:
 *
 *     abs_eb = eb * max(|data|)        (then applied as a plain ABS bound)
 *
 * That bounds `|error| / max(|x|)`, NOT `|error| / |x|`. The two coincide only
 * for elements at the peak magnitude. An element at 1% of peak magnitude sees an
 * effective relative error 100x looser than the number you asked for, and
 * elements near zero are unbounded in relative terms.
 *
 * `QuantizerStage` with `REL` does honour the per-element bound exactly, via
 * log-space quantization (Liang et al., CLUSTER'18; PFPL / LC framework).
 *
 * This program runs the same data through each mode and reports, per mode:
 *   - resolved absolute bound, compressed size / ratio, PSNR, max abs error
 *   - the *pointwise relative error* profile, bucketed by element magnitude
 *
 * The magnitude-bucketed table is the point. Read the `max |e|/|x|` column
 * down the decades: for ABS/NOA/PREL it grows by ~10x per decade as |x| shrinks,
 * for REL it stays flat at <= eb. That growth is exactly the guarantee you do
 * not get from a predictor-fused stage.
 *
 * Usage:
 *   ./eb_mode_analysis <file.f32> --dims <x> [<y> [<z>]] [options]
 *
 *   --eb <val>       Relative error bound for NOA/PREL/REL (default: 1e-3)
 *   --abs-eb <val>   Absolute bound for the ABS run. Default: eb * value_range,
 *                    which makes the ABS run identical to NOA (a useful check).
 *   --radius <n>     Quantization radius for the uint16 modes (default: 4096)
 *   --skip-rel       Skip the QuantizerStage REL run (it needs uint32 codes)
 *   --skip-log       Skip the LogTransform -> Lorenzo -> Quantizer(ABS) run
 *
 * Example:
 *   ./eb_mode_analysis data/CLDHGH.f32 --dims 3600 1800 --eb 1e-3
 *
 * Build:
 *   cmake --preset release -DBUILD_EXAMPLES=ON
 *   cmake --build build/release -j$(nproc)
 *   Binary: build/release/bin/examples/eb_mode_analysis
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <fstream>
#include <limits>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace fz;

#define CUDA_CHECK(expr) do {                                               \
    cudaError_t _e = (expr);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d - %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        std::exit(1);                                                       \
    }                                                                       \
} while (0)

static constexpr float  POOL_MULT   = 4.0f;
static constexpr float  OUTLIER_CAP = 0.20f;

// Magnitude buckets, expressed as |x| / max(|x|). Index 0 is exact zeros;
// bucket k (k >= 1) holds elements with |x|/peak in [10^-k, 10^-(k-1)).
static constexpr int kNumDecades = 8;

struct Stats {
    double vmin = 0.0, vmax = 0.0, max_abs = 0.0;
    double range() const { return vmax - vmin; }
};

struct BucketRow {
    size_t count       = 0;
    double max_rel_err = 0.0;   // max |e| / |x| within the bucket
    size_t violations  = 0;     // elements exceeding the requested relative eb
};

struct ModeResult {
    std::string name;
    std::string guarantee;      // what the mode actually promises
    double abs_eb          = 0.0;
    size_t compressed_size = 0;
    double ratio           = 0.0;
    double max_abs_err     = 0.0;
    double psnr            = 0.0;
    double max_rel_err     = 0.0;
    size_t total_violations = 0;
    size_t zeros_skipped    = 0;
    BucketRow buckets[kNumDecades + 1];
};

// ── Data loading ─────────────────────────────────────────────────────────────

static bool load_data(const char* path, size_t n, std::vector<float>& out) {
    out.resize(n);
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; return false; }
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    if (f.gcount() != static_cast<std::streamsize>(n * sizeof(float))) {
        std::cerr << "Read mismatch: expected " << n * sizeof(float)
                  << " bytes, got " << f.gcount() << "\n";
        return false;
    }
    return true;
}

static Stats compute_stats(const std::vector<float>& v) {
    Stats s;
    s.vmin = s.vmax = v.empty() ? 0.0 : v[0];
    for (float x : v) {
        if (x < s.vmin) s.vmin = x;
        if (x > s.vmax) s.vmax = x;
        const double a = std::fabs(static_cast<double>(x));
        if (a > s.max_abs) s.max_abs = a;
    }
    return s;
}

// ── Error analysis ───────────────────────────────────────────────────────────

/**
 * Bucket every element by |x| / peak and record the worst relative error seen
 * in each bucket. Exact zeros go to bucket 0 and are excluded from the relative
 * statistics entirely (|e|/|x| is undefined there) — they are counted
 * separately so the table still accounts for every element.
 */
static void analyze_errors(
    const std::vector<float>& orig, const std::vector<float>& recon,
    double peak, double rel_eb, ModeResult& r)
{
    const size_t n = orig.size();
    double sq_err = 0.0;

    for (size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(orig[i]);
        const double e = std::fabs(x - static_cast<double>(recon[i]));

        sq_err += e * e;
        if (e > r.max_abs_err) r.max_abs_err = e;

        const double ax = std::fabs(x);
        if (ax == 0.0) {
            r.buckets[0].count++;
            r.zeros_skipped++;
            continue;
        }

        // Decade index: 1 for [0.1, 1] x peak, 2 for [0.01, 0.1) x peak, ...
        const double ratio = ax / peak;
        int k = 1 + static_cast<int>(std::floor(-std::log10(ratio)));
        if (k < 1) k = 1;
        if (k > kNumDecades) k = kNumDecades;

        // Slack on the comparison, in float32 ULP. A stage that enforces its
        // bound in float32 can still land up to ~1 ULP of x past it once the
        // reconstruction is rounded to the nearest float, and this analysis
        // runs in double so it sees that gap. Counting it would libel a mode
        // that is honouring its guarantee as well as float32 permits.
        // Anything genuinely broken (PREL on small |x|) overshoots by orders
        // of magnitude and is unaffected by this slack.
        const double kUlpSlack = 2.0 * static_cast<double>(
            std::numeric_limits<float>::epsilon());

        const double rel = e / ax;
        BucketRow& b = r.buckets[k];
        b.count++;
        if (rel > b.max_rel_err) b.max_rel_err = rel;
        if (rel > rel_eb + kUlpSlack) { b.violations++; r.total_violations++; }
        if (rel > r.max_rel_err) r.max_rel_err = rel;
    }

    const double mse = sq_err / static_cast<double>(n);
    // PSNR against the data range, the SZ-family convention.
    const double rng = peak > 0.0 ? 2.0 * peak : 1.0;
    r.psnr = (mse > 0.0) ? 10.0 * std::log10((rng * rng) / mse)
                         : std::numeric_limits<double>::infinity();
}

// ── Pipeline runners ─────────────────────────────────────────────────────────

/// Lorenzo + Huffman. Handles ABS / NOA / PREL — all of which resolve to one
/// absolute bound, which is precisely why none of them can guarantee REL.
static ModeResult run_lorenzo(
    const std::vector<float>& h_in, const std::array<size_t, 3>& dims,
    float eb, ErrorBoundMode mode, int radius,
    double peak, double rel_eb_for_report,
    const std::string& name, const std::string& guarantee)
{
    ModeResult r;
    r.name      = name;
    r.guarantee = guarantee;

    const size_t n     = h_in.size();
    const size_t bytes = n * sizeof(float);

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    Pipeline p(bytes, MemoryStrategy::PREALLOCATE, POOL_MULT);
    p.setDims(dims[0], dims[1], dims[2]);

    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(eb);
    lq->setErrorBoundMode(mode);
    lq->setQuantRadius(static_cast<uint16_t>(radius));
    lq->setOutlierCapacity(OUTLIER_CAP);
    lq->setZigzagCodes(true);

    auto* huf = p.addStage<HuffmanStage<uint16_t>>();
    huf->setBklen(2 * radius);
    p.connect(huf, lq, "codes");
    p.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    p.compress(d_in, bytes, &d_comp, &comp_sz, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    void*  d_dec   = nullptr;
    size_t dec_sz  = 0;
    p.decompress(d_comp, comp_sz, &d_dec, &dec_sz, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_dec, bytes, cudaMemcpyDeviceToHost));

    r.compressed_size = comp_sz;
    r.ratio           = static_cast<double>(bytes) / static_cast<double>(comp_sz);
    // The stage resolved the user eb into an absolute bound; report it so the
    // "same eb, different meaning" story is visible.
    r.abs_eb          = static_cast<double>(lq->getComputedAbsErrorBound());

    analyze_errors(h_in, h_out, peak, rel_eb_for_report, r);

    cudaFree(d_in);
    return r;
}

/// Quantizer + Bitpack in exact REL mode (log-space codes need uint32).
static ModeResult run_quantizer_rel(
    const std::vector<float>& h_in, const std::array<size_t, 3>& dims,
    float eb, double peak)
{
    ModeResult r;
    r.name      = "REL (Quantizer)";
    r.guarantee = "|e|/|x| <= eb, every element";

    const size_t n     = h_in.size();
    const size_t bytes = n * sizeof(float);

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    Pipeline p(bytes, MemoryStrategy::PREALLOCATE, POOL_MULT);
    p.setDims(dims[0], dims[1], dims[2]);

    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(eb);
    q->setErrorBoundMode(ErrorBoundMode::REL);
    q->setOutlierCapacity(OUTLIER_CAP);

    auto* bp = p.addStage<BitpackStage<uint32_t>>();
    p.connect(bp, q, "codes");
    p.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    p.compress(d_in, bytes, &d_comp, &comp_sz, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    p.decompress(d_comp, comp_sz, &d_dec, &dec_sz, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_dec, bytes, cudaMemcpyDeviceToHost));

    r.compressed_size = comp_sz;
    r.ratio           = static_cast<double>(bytes) / static_cast<double>(comp_sz);
    r.abs_eb          = 0.0;   // no single absolute bound exists in REL mode

    analyze_errors(h_in, h_out, peak, static_cast<double>(eb), r);

    cudaFree(d_in);
    return r;
}

/// LogTransform -> LorenzoQuant(ABS) -> Huffman: the Liang et al. CLUSTER'18
/// scheme. The log transform converts the per-element relative bound into an
/// absolute one *before* the predictor, so an ordinary ABS pipeline delivers
/// the relative guarantee AND still gets to decorrelate.
static ModeResult run_log_lorenzo(
    const std::vector<float>& h_in, const std::array<size_t, 3>& dims,
    float eb, int radius, double peak)
{
    ModeResult r;
    r.name      = "LOG+Lorenzo";
    r.guarantee = "|e|/|x| <= eb, every element";

    const size_t n     = h_in.size();
    const size_t bytes = n * sizeof(float);

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    Pipeline p(bytes, MemoryStrategy::PREALLOCATE, POOL_MULT);
    p.setDims(dims[0], dims[1], dims[2]);

    auto* lg = p.addStage<LogTransformStage<float>>();
    lg->setErrorBound(eb);
    lg->setOutlierCapacity(OUTLIER_CAP);

    // The whole point: the quantizer's ABS bound is log2(1 + eb), not eb.
    // Getting this wrong is silent, so read it off the stage rather than
    // recomputing it here.
    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(lg->quantizerErrorBound());
    lq->setErrorBoundMode(ErrorBoundMode::ABS);
    lq->setQuantRadius(static_cast<uint16_t>(radius));
    lq->setOutlierCapacity(OUTLIER_CAP);
    lq->setZigzagCodes(true);
    p.connect(lq, lg, "output");

    auto* huf = p.addStage<HuffmanStage<uint16_t>>();
    huf->setBklen(2 * radius);
    p.connect(huf, lq, "codes");
    p.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    p.compress(d_in, bytes, &d_comp, &comp_sz, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    p.decompress(d_comp, comp_sz, &d_dec, &dec_sz, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_dec, bytes, cudaMemcpyDeviceToHost));

    r.compressed_size = comp_sz;
    r.ratio           = static_cast<double>(bytes) / static_cast<double>(comp_sz);
    r.abs_eb          = static_cast<double>(lg->quantizerErrorBound());

    analyze_errors(h_in, h_out, peak, static_cast<double>(eb), r);

    cudaFree(d_in);
    return r;
}

// ── Reporting ────────────────────────────────────────────────────────────────

static void print_summary(const std::vector<ModeResult>& rs, double eb) {
    std::cout << "\n"
              << "=================================================================\n"
              << " Summary  (requested eb = " << eb << ")\n"
              << "=================================================================\n";
    std::cout << std::left << std::setw(18) << "mode"
              << std::right
              << std::setw(12) << "abs_eb"
              << std::setw(9)  << "ratio"
              << std::setw(9)  << "PSNR"
              << std::setw(12) << "max |e|"
              << std::setw(13) << "max |e|/|x|"
              << std::setw(12) << "violations" << "\n";
    std::cout << std::string(85, '-') << "\n";

    for (const ModeResult& r : rs) {
        std::cout << std::left << std::setw(18) << r.name << std::right
                  << std::setw(12) << std::scientific << std::setprecision(2)
                  << (r.abs_eb > 0.0 ? r.abs_eb : 0.0)
                  << std::setw(9)  << std::fixed << std::setprecision(2) << r.ratio
                  << std::setw(9)  << std::fixed << std::setprecision(2) << r.psnr
                  << std::setw(12) << std::scientific << std::setprecision(2) << r.max_abs_err
                  << std::setw(13) << std::scientific << std::setprecision(2) << r.max_rel_err
                  << std::setw(12) << r.total_violations
                  << "\n";
    }
    std::cout << "\n'violations' = elements whose |e|/|x| exceeds the requested eb.\n"
              << "Only REL is expected to report zero.\n";
}

static void print_buckets(const ModeResult& r, double eb) {
    std::cout << "\n-- " << r.name << " --  guarantee: " << r.guarantee << "\n";
    std::cout << std::left << std::setw(22) << "  |x| / peak"
              << std::right
              << std::setw(14) << "count"
              << std::setw(15) << "max |e|/|x|"
              << std::setw(14) << "vs eb"
              << std::setw(14) << "violations" << "\n";

    for (int k = 1; k <= kNumDecades; ++k) {
        const BucketRow& b = r.buckets[k];
        if (b.count == 0) continue;

        char label[48];
        std::snprintf(label, sizeof(label), "  [1e-%d, 1e-%d)", k, k - 1);

        const double factor = (eb > 0.0) ? b.max_rel_err / eb : 0.0;
        char factor_s[32];
        std::snprintf(factor_s, sizeof(factor_s), "%.1fx", factor);

        std::cout << std::left << std::setw(22) << label << std::right
                  << std::setw(14) << b.count
                  << std::setw(15) << std::scientific << std::setprecision(2) << b.max_rel_err
                  << std::setw(14) << factor_s
                  << std::setw(14) << b.violations
                  << "\n";
    }
    if (r.zeros_skipped) {
        std::cout << std::left << std::setw(22) << "  exactly 0" << std::right
                  << std::setw(14) << r.zeros_skipped
                  << std::setw(15) << "n/a"
                  << std::setw(14) << "n/a"
                  << std::setw(14) << "n/a" << "\n";
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0]
                  << " <file.f32> --dims <x> [<y> [<z>]] [--eb <v>] "
                     "[--abs-eb <v>] [--radius <n>] [--skip-rel] [--skip-log]\n";
        return 1;
    }

    const std::string path = argv[1];
    std::array<size_t, 3> dims = {0, 1, 1};
    float eb       = 1e-3f;
    float abs_eb   = 0.0f;      // 0 = derive from value range
    int   radius   = 4096;
    bool  skip_rel = false;
    bool  skip_log = false;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << what << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--dims") {
            dims[0] = std::stoull(next("--dims"));
            if (i + 1 < argc && argv[i + 1][0] != '-') dims[1] = std::stoull(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') dims[2] = std::stoull(argv[++i]);
        } else if (a == "--eb")       { eb       = std::stof(next("--eb"));
        } else if (a == "--abs-eb")   { abs_eb   = std::stof(next("--abs-eb"));
        } else if (a == "--radius")   { radius   = std::stoi(next("--radius"));
        } else if (a == "--skip-rel") { skip_rel = true;
        } else if (a == "--skip-log") { skip_log = true;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return 1;
        }
    }

    if (dims[0] == 0) { std::cerr << "--dims is required\n"; return 1; }
    const size_t n = dims[0] * dims[1] * dims[2];

    std::vector<float> h_in;
    if (!load_data(path.c_str(), n, h_in)) return 1;

    const Stats st = compute_stats(h_in);
    if (st.max_abs == 0.0) {
        std::cerr << "Data is all zeros; nothing to analyze.\n";
        return 1;
    }
    if (abs_eb <= 0.0f) abs_eb = static_cast<float>(eb * st.range());

    std::cout << "File            : " << path << "\n"
              << "Elements        : " << n << "  (" << dims[0] << " x "
              << dims[1] << " x " << dims[2] << ")\n"
              << "Value range     : [" << st.vmin << ", " << st.vmax << "]"
              << "  range = " << st.range() << "\n"
              << "Peak |x|        : " << st.max_abs << "\n"
              << "Requested eb    : " << eb << "\n"
              << "ABS run uses    : " << abs_eb << "\n";

    std::vector<ModeResult> results;

    results.push_back(run_lorenzo(
        h_in, dims, abs_eb, ErrorBoundMode::ABS, radius,
        st.max_abs, static_cast<double>(eb),
        "ABS (Lorenzo)", "|e| <= abs_eb"));

    results.push_back(run_lorenzo(
        h_in, dims, eb, ErrorBoundMode::NOA, radius,
        st.max_abs, static_cast<double>(eb),
        "NOA (Lorenzo)", "|e| <= eb * (max-min)"));

    results.push_back(run_lorenzo(
        h_in, dims, eb, ErrorBoundMode::PREL, radius,
        st.max_abs, static_cast<double>(eb),
        "PREL (Lorenzo)", "|e| <= eb * max|x|  -- NOT per-element"));

    if (!skip_rel) {
        results.push_back(run_quantizer_rel(h_in, dims, eb, st.max_abs));
    }
    if (!skip_log) {
        results.push_back(run_log_lorenzo(h_in, dims, eb, radius, st.max_abs));
    }

    print_summary(results, static_cast<double>(eb));

    std::cout << "\n"
              << "=================================================================\n"
              << " Pointwise relative error by element magnitude\n"
              << "=================================================================\n"
              << "Each row is a decade of |x| relative to peak |x|. 'vs eb' is the\n"
              << "worst relative error in that decade as a multiple of the eb you\n"
              << "asked for. For PREL it should roughly track the decade itself.\n";

    for (const ModeResult& r : results) print_buckets(r, static_cast<double>(eb));

    std::cout << "\nTakeaway: PREL's effective per-element relative error degrades\n"
              << "in proportion to how far below peak magnitude an element sits.\n"
              << "If your field spans decades and you need every element accurate\n"
              << "to a relative tolerance, PREL will not deliver it -- use\n"
              << "QuantizerStage REL, and see docs/stages/quantizer.md.\n"
              << "\nNote on the REL row's compression ratio: QuantizerStage REL\n"
              << "quantizes raw values, with no predictor in front, so its\n"
              << "log-bin codes carry the full spatial redundancy of the field\n"
              << "and compress poorly. That is the cost of the exact guarantee\n"
              << "as currently built, not a property of log-space quantization\n"
              << "itself. The LOG+Lorenzo row is exactly that fix: the log\n"
              << "transform placed *upstream* of the predictor (Liang et al.,\n"
              << "CLUSTER'18), so the same guarantee comes with decorrelation.\n";

    return 0;
}
