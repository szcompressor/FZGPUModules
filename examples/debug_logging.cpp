/**
 * debug_logging.cpp — debugging and logging tools for pipeline development.
 *
 * Covers the four main tools for understanding what the pipeline is doing:
 *
 *   1. Logger  — route library log output to stderr or a custom callback.
 *                Log levels: TRACE (0) < DEBUG (1) < INFO (2) < WARN (3) < SILENT.
 *                Default: INFO at runtime, compile-time floor set by FZ_LOG_MIN_LEVEL.
 *
 *   2. printPipeline() / printDAG() / printBufferLifetimes()
 *                Explicit diagnostic dumps — always produce output regardless of
 *                log level. Route through the Logger callback if one is set,
 *                otherwise write to stdout.
 *
 *   3. enableProfiling() + getLastPerfResult()
 *                CUDA-event per-stage timing. Use to find which stage dominates,
 *                compare pipeline variants, or profile before optimising.
 *
 *   4. enableBoundsCheck()
 *                Runtime check that no stage writes beyond its allocated buffer.
 *                Enabled automatically in debug builds; opt-in in release.
 *
 * No external data files required.
 *
 * Usage:
 *   ./build/bin/debug_logging
 *
 * To see TRACE + DEBUG logs, rebuild with:
 *   cmake -DFZ_LOG_MIN_LEVEL=0 --preset release && cmake --build --preset release
 */

#include "fzgpumodules.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
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
    lorenzo->setErrorBoundMode(ErrorBoundMode::REL);
    auto* rle = p.addStage<RLEStage<uint16_t>>();
    p.connect(rle, lorenzo, "codes");
    p.finalize();
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    const std::vector<float> h_data = make_data(N);
    const size_t data_bytes = N * sizeof(float);

    float* d_input = nullptr;
    cudaMalloc(&d_input, data_bytes);
    cudaMemcpy(d_input, h_data.data(), data_bytes, cudaMemcpyHostToDevice);

    // ── Section 1: Logger ─────────────────────────────────────────────────────
    //
    // By default the library is silent — no callback is set. The simplest way
    // to see INFO-level output is Logger::enableStderr(). For finer control,
    // set a custom callback and filter however you like.
    //
    // Log levels that can appear at runtime (subject to compile-time floor):
    //   INFO  — finalize, compress, decompress milestones
    //   WARN  — recoverable unexpected conditions (outlier overflow, etc.)
    //   DEBUG — buffer allocation, pool sizing, internal statistics
    //   TRACE — per-chunk, per-stage kernel execution details (very verbose)
    //
    // The compile-time floor (FZ_LOG_MIN_LEVEL, default INFO=2) gates which
    // FZ_LOG calls are compiled in at all. Setting runtime level below the
    // compile-time floor has no effect — those call sites are already gone.
    std::printf("── Section 1: Logger ────────────────────────────────────────────────\n");
    {
        // Route INFO+ to stderr with the built-in helper.
        Logger::enableStderr(LogLevel::INFO);

        Pipeline p(data_bytes);
        build_pipeline(p);   // finalize() emits INFO logs

        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        p.compress(d_input, data_bytes, &d_comp, &comp_sz);  // compress emits INFO
        cudaDeviceSynchronize();
        std::printf("  compressed to %zu bytes\n", comp_sz);

        // Silence the library again before the next section.
        Logger::setCallback(nullptr);
    }

    // ── Section 2: Custom callback ────────────────────────────────────────────
    //
    // A custom callback lets you filter, tag, or redirect log output
    // (e.g., into a test buffer, a file, or a GUI log panel).
    std::printf("\n── Section 2: custom log callback ───────────────────────────────────\n");
    {
        // Collect log lines into a vector for inspection.
        static std::vector<std::string> log_lines;
        log_lines.clear();

        Logger::setMinLevel(LogLevel::INFO);
        Logger::setCallback([](LogLevel level, const char* msg) {
            const char* tag = Logger::levelTag(level);
            char line[1024];
            snprintf(line, sizeof(line), "[fzgmod:%s] %s", tag, msg);
            log_lines.push_back(line);
        });

        Pipeline p(data_bytes);
        build_pipeline(p);

        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        p.compress(d_input, data_bytes, &d_comp, &comp_sz);
        cudaDeviceSynchronize();

        Logger::setCallback(nullptr);

        std::printf("  captured %zu log lines:\n", log_lines.size());
        for (const auto& line : log_lines)
            std::printf("    %s\n", line.c_str());
    }

    // ── Section 3: printPipeline / printDAG / printBufferLifetimes ────────────
    //
    // These diagnostic helpers produce output unconditionally — they are not
    // filtered by log level. They route through the Logger callback if one is
    // set, or fall back to stdout.
    //
    // printPipeline()       high-level summary: stages, strategies, pool size
    // getDAG()->printDAG()  full DAG: nodes, edges, buffer IDs, level structure
    // getDAG()->printBufferLifetimes()  first/last-use level for each buffer
    //                       (helps understand when pool memory is reclaimed)
    std::printf("\n── Section 3: printPipeline / printDAG / printBufferLifetimes ────────\n");
    {
        Pipeline p(data_bytes);
        build_pipeline(p);

        std::printf("\n[printPipeline]\n");
        p.printPipeline();

        std::printf("\n[printDAG]\n");
        p.getDAG()->printDAG();

        std::printf("\n[printBufferLifetimes]\n");
        p.getDAG()->printBufferLifetimes();
    }

    // ── Section 4: enableProfiling + getLastPerfResult ────────────────────────
    //
    // enableProfiling(true) wraps each stage's execute() with paired CUDA events
    // so you get per-stage GPU time after every compress()/decompress() call.
    //
    // Two throughput views:
    //   dag_elapsed_ms      GPU compute only, excludes host setup.
    //                       Use this for comparing kernels and pipeline variants.
    //   host_elapsed_ms     End-to-end wall time including host overhead.
    //                       Use this for benchmarking total call latency.
    //
    // perf.print() emits a formatted per-stage table to std::cout or a stream.
    std::printf("\n── Section 4: enableProfiling + getLastPerfResult ───────────────────\n");
    {
        Pipeline p(data_bytes);
        build_pipeline(p);
        p.enableProfiling(true);

        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        p.compress(d_input, data_bytes, &d_comp, &comp_sz);
        cudaDeviceSynchronize();

        const auto& perf = p.getLastPerfResult();
        std::printf("\n[compress pass]\n");
        perf.print(std::cout);

        // Per-stage breakdown for custom reporting.
        std::printf("\n[per-stage breakdown]\n");
        for (const auto& s : perf.stages) {
            std::printf("  %-20s  level=%-2d  %.3f ms  %.2f GB/s\n",
                        s.name.c_str(), s.level,
                        s.elapsed_ms, s.throughput_gbs());
        }

        // Decompress to see the decompress pass too.
        void*  d_dec  = nullptr;
        size_t dec_sz = 0;
        p.decompress(d_comp, comp_sz, &d_dec, &dec_sz);
        cudaDeviceSynchronize();

        std::printf("\n[decompress pass]\n");
        p.getLastPerfResult().print(std::cout);
    }

    // ── Section 5: enableBoundsCheck ──────────────────────────────────────────
    //
    // enableBoundsCheck(true) adds a post-execute check after each stage:
    // if a stage's reported actual output size exceeds its allocated buffer,
    // a std::runtime_error is thrown immediately (rather than silently corrupting
    // adjacent memory).
    //
    // This is always on in debug builds (no -DNDEBUG). In release builds it is
    // opt-in — enable it when developing a new stage or chasing a corruption bug.
    // Cost is negligible: one map lookup + one integer compare per output per stage.
    std::printf("\n── Section 5: enableBoundsCheck ─────────────────────────────────────\n");
    {
        Pipeline p(data_bytes);
        build_pipeline(p);
        p.enableBoundsCheck(true);

        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        p.compress(d_input, data_bytes, &d_comp, &comp_sz);
        cudaDeviceSynchronize();
        std::printf("  compress with bounds check: %zu bytes  (no overwrite detected)\n",
                    comp_sz);

        void*  d_dec  = nullptr;
        size_t dec_sz = 0;
        p.decompress(d_comp, comp_sz, &d_dec, &dec_sz);
        cudaDeviceSynchronize();
        std::printf("  decompress with bounds check: %zu bytes\n", dec_sz);
    }

    cudaFree(d_input);
    std::printf("\nDone.\n");
    return 0;
}
