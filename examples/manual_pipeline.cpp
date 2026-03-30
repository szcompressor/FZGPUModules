/**
 * manual_pipeline.cpp — PFPL manual stages vs Pipeline DAG throughput comparison.
 *
 * Runs the PFPL core chain (Quantizer → Diff → Bitshuffle → RZE) two ways:
 *
 *   MANUAL  — stages created and called directly via Stage::execute(), with
 *             pre-allocated intermediate device buffers.
 *   DAG     — same stages wired through the Pipeline builder API and run via
 *             compress().
 *
 * The final concat/header step is intentionally omitted from both paths to
 * isolate the core computation.  All device allocations happen before the
 * timed region.  Throughput is reported for the original (unpadded) data size.
 *
 * Usage:
 *   ./build/bin/pfpl_manual_vs_dag [runs]
 *
 * Example:
 *   ./build/bin/pfpl_manual_vs_dag 20
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace fz;

static const char*      DATA_PATH    = "/home/skyler/data/SDRB/CESM_ATM_1800x3600/CLDHGH.f32";
static constexpr size_t DIM_X        = 3600;
static constexpr size_t DIM_Y        = 1800;
static constexpr size_t N_ELEMS      = DIM_X * DIM_Y;
static constexpr size_t CHUNK        = 16384;   // bytes; all chunked stages use this
static constexpr float  EB           = 1e-4f;
static constexpr float  POOL_MULT    = 3.0f;
static constexpr int    DEFAULT_RUNS = 20;

// ── Helpers ───────────────────────────────────────────────────────────────────

static float* load_to_device() {
    std::vector<float> h(N_ELEMS);
    std::FILE* fp = std::fopen(DATA_PATH, "rb");
    if (!fp) { std::cerr << "[ERROR] cannot open " << DATA_PATH << "\n"; std::exit(1); }
    const size_t got = std::fread(h.data(), sizeof(float), N_ELEMS, fp);
    std::fclose(fp);
    if (got != N_ELEMS) { std::cerr << "[ERROR] short read\n"; std::exit(1); }
    float* d = nullptr;
    cudaMalloc(&d, N_ELEMS * sizeof(float));
    cudaMemcpy(d, h.data(), N_ELEMS * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static float tput_gbs(size_t bytes, double ms) {
    return static_cast<float>(bytes) / (ms * 1e-3) / 1e9f;
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    int runs = DEFAULT_RUNS;
    if (argc > 1) {
        runs = std::stoi(argv[1]);
        if (runs <= 0) { std::cerr << "runs must be > 0\n"; return 1; }
    }

    const size_t data_bytes = N_ELEMS * sizeof(float);

    // Diff, Bitshuffle, and RZE require input aligned to CHUNK bytes.
    // Pipeline handles this transparently; manual path does it explicitly.
    const size_t padded_bytes = (data_bytes + CHUNK - 1) / CHUNK * CHUNK;

    std::cout << "=== PFPL manual stages vs Pipeline DAG — throughput comparison ===\n"
              << "  Dataset : CESM ATM CLDHGH " << DIM_X << "x" << DIM_Y
              << "  (" << std::fixed << std::setprecision(2)
              << data_bytes / (1024.0 * 1024.0) << " MB unpadded, "
              << padded_bytes / (1024.0 * 1024.0) << " MB padded)\n"
              << "  Stages  : Quantizer(ABS,inplace) → Diff → Bitshuffle → RZE\n"
              << "  EB      : " << std::scientific << std::setprecision(1) << EB << "\n"
              << "  Chunk   : " << CHUNK << " bytes\n"
              << "  Runs    : " << runs << " (post-warmup)\n\n";

    float* d_input = load_to_device();
    std::cout << "Loaded " << std::fixed << std::setprecision(2)
              << data_bytes / (1024.0 * 1024.0) << " MB to device.\n\n";

    // =========================================================================
    //  MANUAL PATH — build stages and pre-allocate all buffers
    //  (no allocations inside the timed region)
    // =========================================================================

    // Pool for RZEStage persistent scratch (allocated on first execute(), reused
    // on every subsequent call).
    MemoryPool manual_pool(MemoryPoolConfig(padded_bytes, POOL_MULT));

    // Stages — configured identically to the DAG path below.
    QuantizerStage<float, uint32_t> m_quant;
    m_quant.setErrorBound(EB);
    m_quant.setErrorBoundMode(ErrorBoundMode::ABS);
    m_quant.setQuantRadius(1 << 22);
    m_quant.setOutlierCapacity(0.05f);
    m_quant.setZigzagCodes(true);
    m_quant.setInplaceOutliers(true);   // single "codes" output, no scatter buffers

    DifferenceStage<int32_t, uint32_t> m_diff;
    m_diff.setChunkSize(CHUNK);

    BitshuffleStage m_bshuf;
    m_bshuf.setBlockSize(CHUNK);
    m_bshuf.setElementWidth(4);

    RZEStage m_rze;
    m_rze.setChunkSize(CHUNK);
    m_rze.setLevels(4);

    // Padded copy of the input so chunked stages see CHUNK-aligned boundaries.
    void* d_padded = nullptr;
    cudaMalloc(&d_padded, padded_bytes);
    cudaMemcpy(d_padded, d_input, data_bytes, cudaMemcpyDeviceToDevice);
    if (padded_bytes > data_bytes)
        cudaMemset(static_cast<uint8_t*>(d_padded) + data_bytes, 0,
                   padded_bytes - data_bytes);

    // Intermediate buffers — each sized to padded_bytes (size-preserving stages).
    void* d_codes = nullptr; cudaMalloc(&d_codes, padded_bytes);
    void* d_diff  = nullptr; cudaMalloc(&d_diff,  padded_bytes);
    void* d_bshuf = nullptr; cudaMalloc(&d_bshuf, padded_bytes);

    // RZE worst-case output: original data + stream header (orig_bytes:4, n_chunks:4,
    // sizes:4*n_chunks).
    const size_t n_chunks = padded_bytes / CHUNK;
    const size_t rze_cap  = padded_bytes + 4 + 4 + 4 * n_chunks;
    void* d_rze = nullptr; cudaMalloc(&d_rze, rze_cap);

    cudaDeviceSynchronize();

    // Helper that runs one manual compress pass.
    auto run_manual = [&]() {
        std::vector<void*> in, out;
        std::vector<size_t> sz;

        in = {d_padded}; out = {d_codes}; sz = {padded_bytes};
        m_quant.execute(0, &manual_pool, in, out, sz);

        in = {d_codes}; out = {d_diff}; sz = {padded_bytes};
        m_diff.execute(0, &manual_pool, in, out, sz);

        in = {d_diff}; out = {d_bshuf}; sz = {padded_bytes};
        m_bshuf.execute(0, &manual_pool, in, out, sz);

        in = {d_bshuf}; out = {d_rze}; sz = {padded_bytes};
        m_rze.execute(0, &manual_pool, in, out, sz);

        cudaDeviceSynchronize();
        m_quant.postStreamSync(0);
        m_rze.postStreamSync(0);
    };

    // =========================================================================
    //  PIPELINE (DAG) PATH — build and finalize (outside timed region)
    // =========================================================================

    Pipeline dag_p(data_bytes, MemoryStrategy::PREALLOCATE, POOL_MULT);
    dag_p.enableProfiling(true);    // enables CUDA-event timing for dag_elapsed_ms

    auto* dq = dag_p.addStage<QuantizerStage<float, uint32_t>>();
    dq->setErrorBound(EB);
    dq->setErrorBoundMode(ErrorBoundMode::ABS);
    dq->setQuantRadius(1 << 22);
    dq->setOutlierCapacity(0.05f);
    dq->setZigzagCodes(true);
    dq->setInplaceOutliers(true);

    auto* dd = dag_p.addStage<DifferenceStage<int32_t, uint32_t>>();
    dd->setChunkSize(CHUNK);
    dag_p.connect(dd, dq, "codes");

    auto* db = dag_p.addStage<BitshuffleStage>();
    db->setBlockSize(CHUNK);
    db->setElementWidth(4);
    dag_p.connect(db, dd);

    auto* dr = dag_p.addStage<RZEStage>();
    dr->setChunkSize(CHUNK);
    dr->setLevels(4);
    dag_p.connect(dr, db);

    dag_p.finalize();

    // ── Warmup — prime CUDA JIT and RZE persistent scratch on both paths ──────
    std::cout << "Warming up...\n";
    run_manual();
    {
        void* d_comp = nullptr; size_t comp_sz = 0;
        dag_p.compress(d_input, data_bytes, &d_comp, &comp_sz, 0);
        cudaDeviceSynchronize();
    }
    std::cout << "Done.\n\n";

    // =========================================================================
    //  BENCHMARK
    // =========================================================================

    std::vector<double> manual_ms_v;
    std::vector<double> dag_host_ms_v;
    std::vector<float>  dag_gpu_ms_v;
    manual_ms_v.reserve(static_cast<size_t>(runs));
    dag_host_ms_v.reserve(static_cast<size_t>(runs));
    dag_gpu_ms_v.reserve(static_cast<size_t>(runs));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  run   manual(ms)  man.GB/s   dag-host(ms)  dh.GB/s   dag-GPU(ms)  dg.GB/s\n";
    std::cout << "  " << std::string(76, '-') << "\n";

    for (int i = 0; i < runs; ++i) {
        // ── Manual ────────────────────────────────────────────────────────────
        const auto t0m = Clock::now();
        run_manual();
        const auto t1m = Clock::now();
        const double mms = elapsed_ms(t0m, t1m);
        manual_ms_v.push_back(mms);

        // ── Pipeline DAG ──────────────────────────────────────────────────────
        void* d_comp = nullptr; size_t comp_sz = 0;
        const auto t0d = Clock::now();
        dag_p.compress(d_input, data_bytes, &d_comp, &comp_sz, 0);
        cudaDeviceSynchronize();
        const auto t1d = Clock::now();
        const double dhms = elapsed_ms(t0d, t1d);
        const float  dgms = dag_p.getLastPerfResult().dag_elapsed_ms;
        dag_host_ms_v.push_back(dhms);
        dag_gpu_ms_v.push_back(dgms);

        std::cout << "  " << std::setw(3) << (i + 1)
                  << "   " << std::setw(9) << std::setprecision(3) << mms
                  << "  " << std::setw(7) << std::setprecision(2) << tput_gbs(data_bytes, mms)
                  << "   " << std::setw(11) << std::setprecision(3) << dhms
                  << "  " << std::setw(7) << std::setprecision(2) << tput_gbs(data_bytes, dhms)
                  << "   " << std::setw(10) << std::setprecision(3) << dgms
                  << "  " << std::setw(7) << std::setprecision(2) << tput_gbs(data_bytes, dgms)
                  << "\n";
    }

    // ── Summary table ─────────────────────────────────────────────────────────
    const auto stats = [](const std::vector<double>& v) {
        const int    n    = static_cast<int>(v.size());
        const double mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
        const double mn   = *std::min_element(v.begin(), v.end());
        const double mx   = *std::max_element(v.begin(), v.end());
        return std::make_tuple(mean, mn, mx);
    };
    const auto statsf = [](const std::vector<float>& v) {
        const int   n    = static_cast<int>(v.size());
        const float mean = std::accumulate(v.begin(), v.end(), 0.0f) / n;
        const float mn   = *std::min_element(v.begin(), v.end());
        const float mx   = *std::max_element(v.begin(), v.end());
        return std::make_tuple(mean, mn, mx);
    };

    auto [mmean, mmin, mmax]   = stats(manual_ms_v);
    auto [dhmean, dhmin, dhmax] = stats(dag_host_ms_v);
    auto [dgmean, dgmin, dgmax] = statsf(dag_gpu_ms_v);

    std::cout << "\n  " << std::string(76, '=') << "\n";
    std::cout << std::left  << std::setw(22) << "  Metric"
              << std::right << std::setw(14) << "Manual"
              << std::setw(16) << "DAG-host"
              << std::setw(14) << "DAG-GPU"
              << "\n";
    std::cout << "  " << std::string(64, '-') << "\n";

    const auto row = [&](const std::string& label,
                         double mv, double dhv, double dgv,
                         const std::string& unit) {
        std::cout << std::fixed << std::setprecision(2)
                  << std::left  << std::setw(22) << ("  " + label)
                  << std::right
                  << std::setw(11) << mv  << "  " << unit
                  << std::setw(11) << dhv << "  " << unit
                  << std::setw(11) << dgv << "  " << unit << "\n";
    };

    row("Mean (ms)",       mmean,                          dhmean,                          dgmean,                          "ms  ");
    row("Min  (ms)",       mmin,                           dhmin,                           dgmin,                           "ms  ");
    row("Max  (ms)",       mmax,                           dhmax,                           dgmax,                           "ms  ");
    row("Throughput mean", tput_gbs(data_bytes, mmean),    tput_gbs(data_bytes, dhmean),    tput_gbs(data_bytes, dgmean),    "GB/s");
    row("Throughput best", tput_gbs(data_bytes, mmin),     tput_gbs(data_bytes, dhmin),     tput_gbs(data_bytes, dgmin),     "GB/s");

    std::cout << "  " << std::string(64, '-') << "\n";
    const double dh_delta = 100.0 * (dhmean - mmean) / mmean;
    const double dg_delta = 100.0 * (dgmean - mmean) / mmean;
    std::cout << std::fixed << std::setprecision(1)
              << "  DAG-host vs manual (mean):  "
              << (dh_delta >= 0 ? "+" : "") << dh_delta << "%\n"
              << "  DAG-GPU  vs manual (mean):  "
              << (dg_delta >= 0 ? "+" : "") << dg_delta << "%"
              << "  (GPU-only; excludes DAG orchestration overhead)\n";

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_input);
    cudaFree(d_padded);
    cudaFree(d_codes);
    cudaFree(d_diff);
    cudaFree(d_bshuf);
    cudaFree(d_rze);

    std::cout << "\nDone.\n";
    return 0;
}
