#pragma once

/**
 * @file config.h
 * @brief TOML-based pipeline configuration file support.
 *
 * Loaded by Pipeline::loadConfig() / Pipeline(path) and written by
 * Pipeline::saveConfig().  This header contains no toml++ types — the
 * dependency is an implementation detail confined to config.cpp.
 *
 * Supported stage `type` values: query registeredStageTypes() (below) or run
 * `fzgmod-cli --list-stages`.  Both read the single stage registry in config.cpp,
 * so they cannot go stale — a prose list here did, silently, omitting GInterp,
 * AdaptiveBitpack, TiledLorenzo, GPULZ, ANS, Huffman, ADM and others long after
 * they shipped.
 *
 * File format: human-readable TOML v1.0
 *
 * Minimal example:
 * @code
 *   [pipeline]
 *   dims   = [3600, 1800, 1]
 *   input_size = 25920000
 *
 *   [[stage]]
 *   name            = "lorenzo"
 *   type            = "LorenzoQuant"
 *   input_type      = "float32"
 *   code_type       = "uint16"
 *   error_bound     = 1e-4
 *   error_bound_mode = "ABS"
 *   quant_radius    = 32768
 *   outlier_capacity = 0.10
 *   zigzag_codes    = true
 *
 *   [[stage]]
 *   name        = "bshuf"
 *   type        = "Bitshuffle"
 *   block_size  = 16384
 *   element_width = 2
 *   inputs = [{ from = "lorenzo", port = "codes" }]
 *
 *   [[stage]]
 *   name       = "rze"
 *   type       = "RZE"
 *   chunk_size = 16384
 *   levels     = 4
 *   inputs = [{ from = "bshuf" }]
 * @endcode
 */

// loadConfig/saveConfig themselves are methods on Pipeline — see
// include/pipeline/compressor.h for those declarations.

#include <string>
#include <vector>

namespace fz {

/**
 * @brief Every stage `type` string loadConfig() accepts, in registry order.
 *
 * Reads the one stage registry that also drives TOML load and save dispatch, so
 * it is correct by construction: adding a stage per the procedure in config.cpp
 * updates this automatically, and no second list can drift out of sync.
 *
 * Exposed because consumers need the *inventory*, not just whatever happened to
 * execute — a downstream benchmark harness invalidating cached results per stage
 * has to know a stage exists even when no current pipeline uses it.
 *
 * @return Stage type names, e.g. {"Lorenzo", "LorenzoQuant", "Quantizer", ...}.
 */
std::vector<std::string> registeredStageTypes();

/// A stage's name paired with a hash of the source that implements it.
struct StageFingerprintInfo {
    std::string name;
    std::string fingerprint;  ///< empty if this build has no fingerprint for it
};

/**
 * @brief Per-stage source fingerprints for THIS build.
 *
 * Each fingerprint is a sha256 (truncated to 16 hex chars) over the stage's own
 * sources plus the transitive closure of its repo-local `#include`s, generated at
 * build time by scripts/gen_stage_fingerprints.py.
 *
 * The transitive part is what makes it useful: stages share infrastructure and
 * include each other, so hashing only a stage's own directory would miss a change
 * to the memory pool or to a transform it inlines. A change to a shared header
 * moves every dependent stage's fingerprint; a change to one kernel moves exactly
 * one.
 *
 * Intended use is cache invalidation: a consumer that recorded these alongside a
 * result can re-run only the entries whose stages have since changed, instead of
 * re-running everything or trusting a stale number. Compare fingerprints for
 * equality only — they carry no ordering.
 *
 * Deliberately conservative: comment and formatting edits move the fingerprint
 * too, because proving an edit is semantically inert is not something a hash can
 * do, and a needless re-run is much cheaper than a wrong cached result.
 *
 * @return One entry per registered stage, in registry order.
 */
std::vector<StageFingerprintInfo> stageFingerprints();

}  // namespace fz

