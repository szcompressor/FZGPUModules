#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace fz {

// ──────────────────────────────────────────────────────────────────────────────
// Per-stage timing result
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Timing and throughput for a single stage in the DAG.
 * Populated by CompressionDAG::collectTimings() and stored in PipelinePerfResult.
 */
struct StageTimingResult {
    std::string name;         ///< Stage name (e.g. "lorenzo", "rle")
    int         level;        ///< DAG execution level (0 = source stages)

    /// GPU execution time measured by CUDA events (milliseconds).
    /// Covers the interval [cudaEventRecord(start) → cudaEventRecord(completion)],
    /// which is the time the stage occupied its CUDA stream.
    float       elapsed_ms;

    size_t      input_bytes;  ///< Total bytes across all input buffers
    size_t      output_bytes; ///< Total bytes across all output buffers

    /// Combined read+write throughput in GB/s (uses elapsed_ms, not host time).
    float throughput_gbs() const noexcept;
};

// ──────────────────────────────────────────────────────────────────────────────
// Per-level timing result
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Aggregate timing for all stages that run concurrently at one DAG level.
 * elapsed_ms is the critical-path duration — the longest stage at the level —
 * because parallel stages overlap on different CUDA streams.
 */
struct LevelTimingResult {
    int   level       = 0;  ///< Level index (0 = sources)
    int   parallelism = 0;  ///< Number of concurrent stages at this level
    float elapsed_ms  = 0.0f; ///< Critical-path duration: max(stage.elapsed_ms) over the level
};

// ──────────────────────────────────────────────────────────────────────────────
// Pipeline-level profiling result
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Complete performance snapshot for one compress() or decompress() call.
 *
 * Obtain via:
 *   pipeline.enableProfiling(true);
 *   pipeline.compress(d_in, n, &d_out, &out_size, stream);
 *   auto& r = pipeline.getLastPerfResult();
 *   r.print();
 *
 * Timing layers:
 *   host_elapsed_ms  — wall-clock time for the full call including host overhead
 *                      such as buffer metadata collection, concat, and any
 *                      pipeline construction (e.g. decompressFromFile setup).
 *                      Useful for end-to-end benchmarking but not throughput.
 *   dag_elapsed_ms   — time spent solely inside dag->execute() (GPU compute, ms)
 *                      i.e. the actual GPU compute excluding all host setup.
 *                      This is the denominator for throughput_gbs().
 *   stage elapsed_ms — per-stage GPU time from paired CUDA events; most accurate
 *                      for isolating individual kernel costs.
 *
 * Throughput is always reported as uncompressed data size / dag_elapsed_ms:
 *   compress:   uncompressed_bytes = input_bytes
 *   decompress: uncompressed_bytes = output_bytes
 */
struct PipelinePerfResult {
    bool   is_compress;     ///< true = compress pass, false = decompress pass
    float  host_elapsed_ms; ///< Total host-side wall time including setup (ms)
    float  dag_elapsed_ms;  ///< GPU compute time only — dag->execute() (ms)
    size_t input_bytes;     ///< Bytes fed into the pipeline
    size_t output_bytes;    ///< Bytes produced by the pipeline

    std::vector<StageTimingResult> stages; ///< Per-stage results in topological order
    std::vector<LevelTimingResult> levels; ///< Per-level aggregates in level order

    /// DAG throughput in GB/s: uncompressed data size divided by dag_elapsed_ms.
    /// Isolates actual GPU compute cost from host-side overhead and file I/O.
    float throughput_gbs() const noexcept;

    /// Pipeline throughput in GB/s: uncompressed data size divided by
    /// host_elapsed_ms.  Reflects end-to-end latency including host setup.
    float pipeline_throughput_gbs() const noexcept;

    /// Pretty-print a timing table to \p os (defaults to std::cout).
    void print(std::ostream& os) const;
};

} // namespace fz
