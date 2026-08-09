#pragma once

#include "backend/api.h"

namespace fz {

/**
 * RAII CUDA-event timer for bracketing DAG device execution.
 *
 * Records a start/stop event pair on a stream around dag->execute() (or a
 * cudaGraphLaunch) so that elapsedMs() returns the *device wall time* of the
 * pipeline — i.e. from when the stream reaches the start marker until every
 * kernel (including those joined back from internal streams at the end of
 * execute()) has finished.  This is the honest device-side latency; it
 * deliberately excludes host setup and PCIe transfers issued outside the
 * bracket.
 *
 * Do not replace this with a host clock around execute(): execute() returns
 * after merely *enqueuing* kernels, so a host timer there measures launch
 * latency, not GPU compute. See docs/codebase_notes.md CN-TIMER-1 — which also
 * covers the corollary, that work outside this bracket is invisible here.
 *
 * When `enabled` is false every operation is a no-op and elapsedMs() returns 0,
 * so the events are only created on the profiling path.  The destructor frees
 * the events, making the timer safe across early returns and exceptions.
 *
 * Usage:
 *   DagEventTimer dag_timer(profiling_enabled_);
 *   dag_timer.recordStart(stream);
 *   dag_->execute(stream);
 *   dag_timer.recordStop(stream);
 *   cudaStreamSynchronize(stream);     // must sync before reading
 *   float dag_ms = dag_timer.elapsedMs();
 */
class DagEventTimer {
 public:
  explicit DagEventTimer(bool enabled) : enabled_(enabled) {
    if (enabled_) {
      cudaEventCreate(&start_);
      cudaEventCreate(&stop_);
    }
  }

  ~DagEventTimer() {
    if (enabled_) {
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
    }
  }

  DagEventTimer(const DagEventTimer&) = delete;
  DagEventTimer& operator=(const DagEventTimer&) = delete;

  void recordStart(cudaStream_t stream) {
    if (enabled_) cudaEventRecord(start_, stream);
  }

  void recordStop(cudaStream_t stream) {
    if (enabled_) cudaEventRecord(stop_, stream);
  }

  /// Device wall time in ms between the start and stop markers.
  /// Only valid after the stream carrying the stop event has been synchronized.
  /// Returns 0 when disabled.
  float elapsedMs() const {
    float ms = 0.0f;
    if (enabled_) cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
  }

  bool enabled() const { return enabled_; }

 private:
  bool enabled_;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

}  // namespace fz
