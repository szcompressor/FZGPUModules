#pragma once

/**
 * @file config.h
 * @brief TOML-based pipeline configuration file support.
 *
 * Loaded by Pipeline::loadConfig() / Pipeline(path) and written by
 * Pipeline::saveConfig().  This header contains no toml++ types — the
 * dependency is an implementation detail confined to config.cpp.
 *
 * Supported stage `type` values:
 *   LorenzoQuant  — fused float predictor+quantizer (cuSZ-style)
 *   Lorenzo       — plain integer delta predictor (lossless, use after Quantizer)
 *   Quantizer     — standalone float-to-integer quantizer
 *   Bitshuffle, RZE, RLE, Bitpack, Difference, Zigzag, Negabinary
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

// No public types to expose — loadConfig/saveConfig are methods on Pipeline.
// See include/pipeline/compressor.h for the API declarations.

