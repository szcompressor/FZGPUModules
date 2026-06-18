# Command Line Interface {#cli_overview}

FZGPUModules provides a fully-featured CLI (fzgmod-cli) for testing, comparing,
and benchmarking pipelines without writing C++ code.

## Dynamic linear pipelines

Chain stages with `--stages` and `->` separators (always quote the stage list to
prevent shell redirection):

```bash
# Lorenzo -> Bitshuffle -> RZE
fzgmod-cli -z -i data.f32 -o compressed.fzm --stages "lorenzo->bitshuffle->rze" -m rel -e 1e-3 --report

# Four-stage pipeline
fzgmod-cli -z -i data.f32 --stages "lorenzo->diff->bitshuffle->rze" -e 1e-4 --report
```

Example Output:
```txt
[Compress Report]
  Input size:      25920000 bytes
  Compressed size: 2138592 bytes
  Ratio:           12.12x
  Time:            2.69 ms
  Throughput:      9.63 GB/s
```

## Decompress, compare, and report

```bash
fzgmod-cli -x -i compressed.fzm -o decompressed.f32 --compare data.f32 --report
# Prints: Output size, Time, Throughput, Value Range, Max Abs Error, PSNR, NRMSE
```

Example Output:
```txt
[Decompress Report]
  Output size:     25920000 bytes
  (Padded size:    25935872 bytes, truncated to match original)
  Time:            505.334 ms
  Throughput:      0.05 GB/s
  Value Range:     [1.64e-03, 8.96e-01] (Span: 8.94e-01)
  Max Abs Error:   8.96e-05
  PSNR:            84.76 dB
  NRMSE:           5.78e-05
```

## Branched pipelines via TOML config

```bash
fzgmod-cli -z -i data.f32 -c examples/presets/pfpl.toml -o compressed.fzm --report
```

See the \ref config_file_overview "Config file reference" page for schema
details and examples.

## Benchmarking

```bash
fzgmod-cli -b -i data.f32 --stages "lorenzo->bitshuffle->rze" -m rel -e 1e-3 --runs 10
```

Example Output:
```txt
[benchmark] compress summary
  runs:           10
  host ms:        mean=1.375
  dag ms:         mean=0.646
  throughput:     18.85 GB/s (host mean)

[benchmark] decompress summary
  runs:           10
  host ms:        mean=1.746
  dag ms:         mean=1.321
  throughput:     14.85 GB/s (host mean)

[Quality Report]
  Input size:      25920000 bytes
  Compressed size: 7116544 bytes
  Ratio:           3.64x
  Value Range:     [1.64e-03, 8.96e-01] (Span: 8.94e-01)
  Max Abs Error:   8.80e-04
  PSNR:            74.96 dB
  NRMSE:           1.79e-04
```

## Machine-readable JSON reports

`--report-json <file>` writes a standalone, structured report to `<file>` for
programmatic consumption (benchmark harnesses, regression dashboards, CI). It works
in all three modes and is independent of the human-readable `--report`: JSON goes to
its own file, while logs, warnings, and the `--profile` table stay on stdout/stderr,
so the two never collide.

```bash
# Compress, emit JSON alongside the human report
fzgmod-cli -z -i data.f32 -l 3600x1800 -m rel -e 1e-3 \
           -o compressed.fzm --report-json report.json

# Benchmark with per-rep timing arrays in the JSON
fzgmod-cli -b -i data.f32 -l 3600x1800 -m rel -e 1e-3 --runs 10 \
           --report-json bench.json

# Decompress with quality metrics
fzgmod-cli -x -i compressed.fzm -o out.f32 --compare data.f32 \
           --report-json decomp.json
```

Passing `--report-json` automatically enables profiling (so device timing and the
per-stage breakdown are populated) without printing the stdout perf table.

Abbreviated `benchmark` output:

```json
{
  "schema_version": "1.0",
  "tool": { "name": "fzgmod-cli", "version": "2.0.0", "git_sha": "abcd123" },
  "status": "ok",
  "error_message": null,
  "config": { "operation": "benchmark", "dtype": "f32", "dims": [3600, 1800, 1],
              "num_elements": 6480000, "error_mode": "rel", "error_bound": 0.001,
              "pipeline": "lorenzo->bitshuffle->rze", "n_runs": 10 },
  "size": { "original_bytes": 25920000, "compressed_bytes": 4980656,
            "ratio": 5.20, "bitrate_bits_per_elem": 6.15 },
  "timing": {
    "n_runs": 10,
    "timing_method": "cuda_events_dag",
    "compress":   { "device_ms":   { "median": 0.61, "min": 0.57, "max": 1.12, "all": [/* per rep */] },
                    "host_wall_ms": { "median": 1.47, "min": 1.21, "max": 2.47, "all": [/* per rep */] } },
    "decompress": { "device_ms":   { "median": 1.37, "min": 1.34, "max": 1.59, "all": [/* per rep */] },
                    "host_wall_ms": { "median": 1.42, "min": 1.39, "max": 3.92, "all": [/* per rep */] } }
  },
  "throughput": { "compress_gbs": 42.1, "decompress_gbs": 18.9,
                  "basis": "original_bytes / device_ms median (GB/s, 1e9)" },
  "memory": { "peak_device_bytes": 139888928, "method": "pipeline_pool_tracker" },
  "quality": { "val_min": -0.81, "val_max": 0.81, "val_range": 1.62,
               "max_abs_err": 0.00081, "psnr_db": 70.79, "nrmse": 0.00029 },
  "stages": [
    { "name": "LorenzoQuant", "phase": "compress", "device_ms": 0.18 },
    { "name": "Bitshuffle",   "phase": "compress", "device_ms": 0.11 },
    { "name": "RZE",          "phase": "compress", "device_ms": 0.50 }
  ]
}
```

Contract notes for consumers:

- **`device_ms` vs `host_wall_ms`.** `device_ms` is true GPU wall time measured by
  CUDA events bracketing the pipeline (`timing_method: cuda_events_dag`); it excludes
  PCIe transfers and host setup. `host_wall_ms` is the full end-to-end call including
  file I/O and pipeline construction. For cross-tool throughput comparison use
  `device_ms`. Never compare this tool's `host_wall_ms` against another tool's
  kernel-only timing — single-shot decompress `host_wall_ms` includes a fresh
  pipeline build and can be 10–100× the device time.
- **Recompute from raw counts.** `original_bytes`, `compressed_bytes`, and
  `num_elements` are always present; `ratio`, `bitrate_bits_per_elem`, and
  `throughput.*` are conveniences derived from them. Throughput is decimal GB/s
  (bytes / 1e9 / seconds). Recompute with your own unit convention if you mix tools.
- **Per-operation shape.** `-z` populates `timing.compress` only; `-x` populates
  `timing.decompress` only; both are single-rep (`n_runs: 1`, `all` has one element).
  `-b` populates both phases with `--runs` reps. `quality` appears only when an
  original is available (`--compare`, or benchmark self-compare). `memory` is absent
  for `-x`. Optional blocks are omitted entirely rather than set to null.
- **Failures.** On error the file still contains valid JSON with `"status": "error"`,
  a non-null `error_message`, and whatever `config` was resolved before the failure;
  the process exits non-zero. Record `status` rather than inferring a crash from a
  missing file.

## Key flags

| Flag | Description |
|---|---|
| -z / -x / -b | Compress / Decompress / Benchmark mode |
| -i <file> | Input file |
| -o <file> | Output file |
| -c <file.toml> | Load pipeline from TOML config |
| --stages <s1->s2->...> | Ordered stage chain (lorenzo, quantizer, bitshuffle, rze, diff, rle, huffman) |
| -t <f32\|f64> | Data type (default: f32) |
| -m <rel\|abs\|noa> | Error bound mode (default: rel) |
| -e <val> | Error bound value (default: 1e-3) |
| -r <val> | Quantization radius (default: 32768) |
| -l <x>x<y>x<z> | Dimensions (inferred if omitted) |
| -R / --report | Print compression ratio and throughput |
| --report-json <file> | Write a machine-readable JSON report to `<file>` (all modes) |
| --compare <file> | Compare decompressed vs original (MaxErr, PSNR, NRMSE) |
| --runs <n> | Benchmark iteration count (default: 10) |
