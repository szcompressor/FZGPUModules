/**
 * huffman_book_drift — does a pinned Huffman codebook decay across a run?
 *
 * `HuffmanBookSource::Adaptive` histograms the *first* compress() call, builds a
 * codebook from it, and reuses that book for every later call.  Every ratio
 * measurement we have so far compressed a single field through a single pipeline,
 * so the pinned book was always fitted to exactly the data being measured — the
 * optimistic bound.  This harness measures the case that bound hides: **one
 * pipeline instance, many successive compress() calls, each on different data.**
 *
 * That is the shape of a real streaming workload (successive timesteps, successive
 * levels of a 3-D field, or a whole directory of variables through one compressor),
 * and it is the scenario in which a pinned book can go stale.
 *
 * Three book sources are run over the identical step sequence:
 *
 *   PerBlock  — rebuilds every call.  The ratio ceiling, and the reference.
 *   Adaptive  — fitted to step 0, then frozen.  Drift shows up as a loss that
 *               grows with step index while PerBlock's does not.
 *   Fixed     — an analytic model book, fitted to nothing.  Included to decide
 *               whether it earns its place at all.
 *
 * PerBlock doubles as the "refit on every call" upper bound, so the gap between it
 * and Adaptive is exactly what a refit policy could recover.
 *
 * Usage:
 *   fzgmod-profile-huffman-drift [options] [file ...]
 *
 *     --dims XxY[xZ]     geometry of ONE step        (default 3600x1800)
 *     --slabs N          split each input file into N equal steps along the
 *                        slowest axis; with one 3-D file this walks its levels
 *     --eb VAL           error bound, NOA mode       (default 1e-4)
 *     --radius R         quantizer radius; bklen = 2R  (default 512)
 *     --scale S          Fixed-book Laplace scale    (default bklen/64)
 *     --refit-interval N Adaptive: refit the codebook every N calls (0 = off)
 *     --skip N           drop the first N steps before running, so the
 *                        Adaptive book is fitted to step N rather than step 0
 *     --csv PATH         also write per-step rows as CSV
 *
 *   Files are steps, in the order given, and must share one geometry.  With no
 *   files, the in-repo data/CLDHGH.f32 is split via --slabs (default 8) so the
 *   harness runs on any checkout — though a single field's slabs drift far less
 *   than genuinely different fields, so prefer passing real files:
 *
 *     fzgmod-profile-huffman-drift --dims 3600x1800 \
 *         /data/SDRB/CESM_ATM_1800x3600/{CLDHGH,CLDTOT,FLNS,LHFLX,FREQSH}.f32
 *
 *     fzgmod-profile-huffman-drift --dims 3600x1800 --slabs 26 \
 *         /data/SDRB/CESM_ATM_26x1800x3600/CLOUD.f32
 *
 * Reading the output: the "vs step 0" column is the drift signal.  If a book mode
 * holds its step-0 loss across every later step, the pinned book is not decaying;
 * if the loss grows monotonically, it is, and a refit policy is worth building.
 *
 * Nsys (optional — this harness is primarily about ratio, not kernel time):
 *   nsys profile --trace=cuda,nvtx -o hufdrift ./fzgmod-profile-huffman-drift
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace fz;
using Clock = std::chrono::high_resolution_clock;

static constexpr float POOL_MULT = 4.0f;

// ── Options ──────────────────────────────────────────────────────────────────

struct Options {
    size_t                   nx = 3600, ny = 1800, nz = 1;
    int                      slabs = 0;         // 0 = use whole file as one step
    float                    eb = 1e-4f;
    int                      radius = 512;
    double                   scale = 0.0;       // 0 = derive from bklen
    int                      skip = 0;          // drop the first N steps
    uint32_t                 refit_interval = 0;
    std::string              csv;
    std::vector<std::string> files;
};

static void die(const std::string& msg) {
    std::cerr << "huffman_book_drift: " << msg << "\n";
    std::exit(1);
}

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value after ") + what);
            return argv[++i];
        };
        if (a == "--dims") {
            const std::string v = next("--dims");
            o.nz = 1;
            if (std::sscanf(v.c_str(), "%zux%zux%zu", &o.nx, &o.ny, &o.nz) < 2 &&
                std::sscanf(v.c_str(), "%zux%zu", &o.nx, &o.ny) < 2)
                die("could not parse --dims \"" + v + "\" (expected XxY or XxYxZ)");
        }
        else if (a == "--slabs")  o.slabs  = std::atoi(next("--slabs").c_str());
        else if (a == "--eb")     o.eb     = std::atof(next("--eb").c_str());
        else if (a == "--radius") o.radius = std::atoi(next("--radius").c_str());
        else if (a == "--scale")  o.scale  = std::atof(next("--scale").c_str());
        else if (a == "--skip")   o.skip   = std::atoi(next("--skip").c_str());
        else if (a == "--refit-interval") o.refit_interval = static_cast<uint32_t>(std::atoi(next("--refit-interval").c_str()));
        else if (a == "--csv")    o.csv    = next("--csv");
        else if (a.rfind("--", 0) == 0) die("unknown option " + a);
        else o.files.push_back(a);
    }
    if (o.files.empty()) {
        o.files.push_back(FZ_DATA_DIR "/CLDHGH.f32");
        if (o.slabs == 0) o.slabs = 8;
        std::cout << "No input files given — falling back to the in-repo sample,\n"
                     "split into " << o.slabs << " slabs.  Slabs of one field drift far "
                     "less than\ndifferent fields do, so treat this as a smoke run.\n\n";
    }
    if (o.radius <= 0) die("--radius must be positive");
    if (o.slabs < 0)   die("--slabs must be >= 0");
    return o;
}

// ── Step loading ─────────────────────────────────────────────────────────────

struct Step {
    std::string        name;
    std::vector<float> host;
};

// Geometry of a single step.  Slabbing cuts along the slowest-varying axis, so a
// 3-D file is split across z and a 2-D one across y.  Getting this wrong would
// silently hand Lorenzo the wrong stride and invalidate every ratio below, so the
// split axis must divide exactly.
struct StepDims { size_t x, y, z; };

static StepDims step_dims(const Options& o) {
    const int n = (o.slabs > 0) ? o.slabs : 1;
    if (n == 1) return {o.nx, o.ny, o.nz};
    if (o.nz > 1) {
        if (o.nz % static_cast<size_t>(n) != 0)
            die("--slabs " + std::to_string(n) + " does not divide z extent " +
                std::to_string(o.nz));
        return {o.nx, o.ny, o.nz / static_cast<size_t>(n)};
    }
    if (o.ny % static_cast<size_t>(n) != 0)
        die("--slabs " + std::to_string(n) + " does not divide y extent " +
            std::to_string(o.ny));
    return {o.nx, o.ny / static_cast<size_t>(n), o.nz};
}

// Reads `count` floats starting at `offset` floats into the file.
static std::vector<float> read_floats(const std::string& path, size_t offset, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open " + path);
    f.seekg(static_cast<std::streamoff>(offset * sizeof(float)), std::ios::beg);
    std::vector<float> v(count);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (static_cast<size_t>(f.gcount()) != count * sizeof(float))
        die(path + ": expected " + std::to_string(count * sizeof(float)) +
            " bytes at offset " + std::to_string(offset * sizeof(float)) +
            " but read " + std::to_string(f.gcount()) +
            " — check --dims / --slabs against the file size");
    return v;
}

static std::vector<Step> load_steps(const Options& o, size_t& step_elems) {
    // A slab is one step; without --slabs the whole file is the step.
    const size_t per_file = o.nx * o.ny * o.nz;
    const int    nslab    = (o.slabs > 0) ? o.slabs : 1;
    if (per_file % static_cast<size_t>(nslab) != 0)
        die("step element count " + std::to_string(per_file) +
            " is not divisible by --slabs " + std::to_string(nslab));
    step_elems = per_file / static_cast<size_t>(nslab);

    std::vector<Step> steps;
    for (const auto& path : o.files) {
        const std::string base = path.substr(path.find_last_of('/') + 1);
        for (int s = 0; s < nslab; ++s) {
            Step st;
            st.name = (nslab == 1) ? base : base + "[" + std::to_string(s) + "]";
            st.host = read_floats(path, static_cast<size_t>(s) * step_elems, step_elems);
            steps.push_back(std::move(st));
        }
    }
    return steps;
}

// ── One measurement ──────────────────────────────────────────────────────────

enum class Mode { PerBlock, Adaptive, Fixed };

static const char* mode_name(Mode m) {
    switch (m) {
        case Mode::Adaptive: return "Adaptive";
        case Mode::Fixed:    return "Fixed";
        default:             return "PerBlock";
    }
}

struct StepResult { size_t comp_bytes; double ms; double max_err; bool decoded; };

// Builds ONE pipeline and pushes every step through it in order — the whole point
// of the harness.  Rebuilding per step would re-fit the book and measure nothing.
static std::vector<StepResult> run_mode(
    Mode mode, const Options& o, const std::vector<Step>& steps,
    size_t step_elems, float* d_input, uint32_t* refits_out = nullptr)
{
    const size_t   step_bytes = step_elems * sizeof(float);
    const uint32_t bklen      = static_cast<uint32_t>(2 * o.radius);

    Pipeline p(step_bytes, MemoryStrategy::PREALLOCATE, POOL_MULT);
    // Dims describe one step, which is the unit actually compressed.
    const StepDims sd = step_dims(o);
    p.setDims(sd.x, sd.y, sd.z);

    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(o.eb);
    lq->setErrorBoundMode(ErrorBoundMode::NOA);
    lq->setQuantRadius(o.radius);
    lq->setOutlierCapacity(0.2f);
    lq->setZigzagCodes(true);

    // Swap this stage (and the connect() port below) to profile a different
    // Huffman consumer; nothing else in the harness is pipeline-specific.
    auto* huf = p.addStage<HuffmanStage<uint16_t>>();
    huf->setBklen(bklen);
    huf->setEncodeMode(HuffmanEncodeMode::Fine);
    if (mode == Mode::Adaptive) {
        huf->setBookSource(HuffmanBookSource::Adaptive);
        huf->setRefitInterval(o.refit_interval);
    } else if (mode == Mode::Fixed) {
        HuffmanBookSpec spec;
        spec.model  = HuffmanBookModel::Laplace;
        spec.center = 0.0;  // zigzag codes cluster at 0
        spec.scale  = (o.scale > 0.0) ? o.scale : bklen / 64.0;
        huf->setFixedBookFromModel(spec);
    }
    p.connect(huf, lq, "codes");
    p.finalize();

    std::vector<StepResult> out;
    out.reserve(steps.size());
    if (refits_out) *refits_out = 0;
    for (const auto& st : steps) {
        cudaMemcpy(d_input, st.host.data(), step_bytes, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        const auto t0 = Clock::now();
        p.compress(d_input, step_bytes, &d_comp, &comp_sz, 0);
        cudaDeviceSynchronize();
        const auto t1 = Clock::now();

        // Always decompress and check the bound.  A compression ratio is
        // meaningless if the stream does not decode, and the degenerate cases this
        // harness deliberately walks into (a near-constant slab, a codebook fitted
        // to one symbol) are exactly where an encoder is most likely to emit
        // something it cannot read back.
        double max_err = 0.0;
        bool   decoded = true;
        void*  d_rec   = nullptr;
        size_t rec_sz  = 0;
        try {
            p.decompress(d_comp, comp_sz, &d_rec, &rec_sz, 0);
            cudaDeviceSynchronize();
        } catch (const std::exception& e) {
            decoded = false;
            std::cerr << "  [decompress failed on " << st.name << ": " << e.what() << "]\n";
        }
        if (decoded && rec_sz == step_bytes) {
            std::vector<float> h_rec(step_elems);
            cudaMemcpy(h_rec.data(), d_rec, step_bytes, cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < step_elems; ++i)
                max_err = std::max(max_err,
                                   std::fabs(static_cast<double>(h_rec[i]) - st.host[i]));
        } else if (decoded) {
            decoded = false;
            std::cerr << "  [decompress size mismatch on " << st.name << ": got "
                      << rec_sz << " want " << step_bytes << "]\n";
        }

        out.push_back({comp_sz,
                       std::chrono::duration<double, std::milli>(t1 - t0).count(),
                       max_err, decoded});
        if (refits_out) *refits_out = huf->getRefitCount();
    }
    return out;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const Options o = parse_args(argc, argv);

    size_t step_elems = 0;
    std::vector<Step> steps = load_steps(o, step_elems);

    // --skip drops leading steps *before* anything runs, so that Adaptive fits its
    // book on the first step that survives. This matters: if step 0 is degenerate
    // (a constant slab), Adaptive pins a book built from a single symbol and stays
    // poisoned for the whole run — real behaviour, but not what you want to measure
    // when the question is ordinary drift.
    if (o.skip > 0) {
        if (static_cast<size_t>(o.skip) >= steps.size())
            die("--skip " + std::to_string(o.skip) + " drops all " +
                std::to_string(steps.size()) + " steps");
        steps.erase(steps.begin(), steps.begin() + o.skip);
        std::cout << "Skipped the first " << o.skip << " step(s).\n";
    }
    const size_t step_bytes = step_elems * sizeof(float);

    const StepDims sd = step_dims(o);
    std::cout << "Steps: " << steps.size()
              << "   step dims: " << sd.x << "x" << sd.y << "x" << sd.z
              << "   step size: " << step_bytes / (1024.0 * 1024.0) << " MB"
              << "   eb " << o.eb << " (NOA)"
              << "   radius " << o.radius << "   bklen " << 2 * o.radius << "\n\n";
    if (steps.size() < 2)
        std::cout << "WARNING: only one step — this harness measures how a pinned book\n"
                     "behaves across *successive different* inputs. Pass more files or "
                     "use --slabs.\n\n";

    float* d_input = nullptr;
    if (cudaMalloc(&d_input, step_bytes) != cudaSuccess) die("cudaMalloc failed");

    const Mode modes[] = {Mode::PerBlock, Mode::Adaptive, Mode::Fixed};
    std::vector<std::vector<StepResult>> res;
    uint32_t adaptive_refits = 0;
    for (Mode m : modes)
        res.push_back(run_mode(m, o, steps, step_elems, d_input,
                               m == Mode::Adaptive ? &adaptive_refits : nullptr));

    // ── Per-step table ───────────────────────────────────────────────────────
    std::cout << std::left << std::setw(26) << "step"
              << std::right << std::setw(10) << "PerBlock"
              << std::setw(10) << "Adaptive" << std::setw(9) << "loss"
              << std::setw(10) << "Fixed"    << std::setw(9) << "loss" << "\n";
    std::cout << std::string(74, '-') << "\n";

    auto cr = [&](size_t comp) { return static_cast<double>(step_bytes) / comp; };

    // NOA bound is relative to each step's own value range, so derive the
    // tolerance per step rather than assuming a global range.
    auto step_range = [&](size_t i) {
        const auto mm = std::minmax_element(steps[i].host.begin(), steps[i].host.end());
        return static_cast<double>(*mm.second - *mm.first);
    };
    std::vector<double> loss_ad, loss_fx;

    // A step whose values are all identical has zero value range, which makes the
    // NOA/PREL derived bound degenerate — such a step is not a valid ratio
    // measurement and must not enter the drift statistics. (Compressing one is also
    // how CESM CLOUD levels 0-2, which are all exactly 0.0, produce a CR of 488.)
    std::vector<bool> degenerate(steps.size(), false);
    int n_degen = 0;
    for (size_t i = 0; i < steps.size(); ++i)
        if (step_range(i) == 0.0) { degenerate[i] = true; ++n_degen; }
    if (n_degen)
        std::cout << "NOTE: " << n_degen << " step(s) are constant (zero value range).\n"
                     "      Marked [const] below and excluded from the drift summary;\n"
                     "      their ratios are not meaningful.\n\n";

    int bad = 0;
    for (size_t i = 0; i < steps.size(); ++i) {
        const double bound_tol = o.eb * step_range(i) * 1.01 + 1e-12;
        const double base = cr(res[0][i].comp_bytes);
        const double ad   = cr(res[1][i].comp_bytes);
        const double fx   = cr(res[2][i].comp_bytes);
        const double la   = 100.0 * (base - ad) / base;
        const double lf   = 100.0 * (base - fx) / base;
        if (!degenerate[i]) {
            loss_ad.push_back(la);
            loss_fx.push_back(lf);
            for (size_t m = 0; m < 3; ++m)
                if (!res[m][i].decoded || res[m][i].max_err > bound_tol) ++bad;
        }

        // Flag anything that failed to decode or violated the error bound; a
        // ratio from such a step is not a result, it is a bug report.
        std::string flag;
        if (degenerate[i]) flag += " [const]";
        for (size_t m = 0; m < 3; ++m) {
            if (degenerate[i]) break;
            if (!res[m][i].decoded)                 flag += std::string(" !") + mode_name(modes[m]) + ":nodecode";
            else if (res[m][i].max_err > bound_tol) flag += std::string(" !") + mode_name(modes[m]) + ":err="
                                                          + std::to_string(res[m][i].max_err);
        }
        std::cout << std::left << std::setw(26) << steps[i].name.substr(0, 25)
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(10) << base
                  << std::setw(10) << ad << std::setw(8) << la << "%"
                  << std::setw(10) << fx << std::setw(8) << lf << "%"
                  << flag << "\n";
    }

    // ── Drift summary ────────────────────────────────────────────────────────
    // Step 0 is the step Adaptive fitted its book to. Comparing it against the
    // mean of the later steps isolates decay from the baseline cost of pinning.
    auto mean_from = [](const std::vector<double>& v, size_t from) {
        if (v.size() <= from) return 0.0;
        return std::accumulate(v.begin() + from, v.end(), 0.0) /
               static_cast<double>(v.size() - from);
    };

    if (bad) {
        std::cout << "\n*** " << bad << " (mode, step) cell(s) failed to decode or "
                     "exceeded the error bound. ***\n"
                     "*** Ratios from those cells are not valid results. ***\n";
    } else {
        std::cout << "\nAll cells decoded within the error bound.\n";
    }

    std::cout << "\nDrift (ratio loss vs PerBlock, negative = better than PerBlock)\n";
    std::cout << std::string(74, '-') << "\n";
    std::cout << std::fixed << std::setprecision(2);
    if (loss_ad.empty()) {
        std::cout << "  (no non-degenerate steps — nothing to report)\n";
        cudaFree(d_input);
        return 1;
    }
    std::cout << "  Adaptive  first non-const step (book source): " << loss_ad[0] << "%\n";
    if (loss_ad.size() > 1) {
        const double later = mean_from(loss_ad, 1);
        const double worst = *std::max_element(loss_ad.begin() + 1, loss_ad.end());
        std::cout << "            steps 1..n mean:         " << later << "%\n"
                  << "            steps 1..n worst:        " << worst << "%\n"
                  << "            DRIFT (later - step 0):  " << later - loss_ad[0] << "%\n";
    }
    std::cout << "  Adaptive  codebook refits:         " << adaptive_refits
              << " (0 = the book was never judged stale)\n";
    std::cout << "  Fixed     mean over all steps:     " << mean_from(loss_fx, 0) << "%\n";

    // Throughput is reported for completeness but is the weaker signal here: one
    // timed call per (mode, step) with no repetition.
    std::cout << "\nCompress GB/s (single timed call per cell — indicative only)\n";
    std::cout << std::string(74, '-') << "\n";
    for (size_t m = 0; m < 3; ++m) {
        double sum = 0.0;
        for (const auto& r : res[m]) sum += (step_bytes / 1e9) / (r.ms / 1e3);
        std::cout << "  " << std::left << std::setw(10) << mode_name(modes[m])
                  << std::right << std::setprecision(2)
                  << sum / static_cast<double>(res[m].size()) << "\n";
    }

    if (!o.csv.empty()) {
        std::ofstream f(o.csv);
        if (!f) die("cannot write " + o.csv);
        f << "step_index,step_name,mode,compressed_bytes,ratio,ms\n";
        for (size_t m = 0; m < 3; ++m)
            for (size_t i = 0; i < steps.size(); ++i)
                f << i << ',' << steps[i].name << ',' << mode_name(modes[m]) << ','
                  << res[m][i].comp_bytes << ',' << cr(res[m][i].comp_bytes) << ','
                  << res[m][i].ms << '\n';
        std::cout << "\nWrote " << o.csv << "\n";
    }

    cudaFree(d_input);
    return 0;
}
