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

## Key flags

| Flag | Description |
|---|---|
| -z / -x / -b | Compress / Decompress / Benchmark mode |
| -i <file> | Input file |
| -o <file> | Output file |
| -c <file.toml> | Load pipeline from TOML config |
| --stages <s1->s2->...> | Ordered stage chain (lorenzo, quantizer, bitshuffle, rze, diff, rle) |
| -t <f32\|f64> | Data type (default: f32) |
| -m <rel\|abs\|noa> | Error bound mode (default: rel) |
| -e <val> | Error bound value (default: 1e-3) |
| -r <val> | Quantization radius (default: 32768) |
| -l <x>x<y>x<z> | Dimensions (inferred if omitted) |
| -R / --report | Print compression ratio and throughput |
| --compare <file> | Compare decompressed vs original (MaxErr, PSNR, NRMSE) |
| --runs <n> | Benchmark iteration count (default: 10) |
