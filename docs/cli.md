# Command Line Interface {#cli_overview}

FZGPUModules provides a fully-featured CLI (fzgmod-cli) for testing, comparing,
and benchmarking pipelines without writing C++ code.

## Dynamic linear pipelines

Chain stages with --stages and -> separators (always quote the stage list to
prevent shell redirection):

```bash
# Lorenzo -> Bitshuffle -> RZE
fzgmod-cli -z -i data.f32 -o compressed.fzm --stages "lorenzo->bitshuffle->rze" -m rel -e 1e-3

# Four-stage pipeline
fzgmod-cli -z -i data.f32 --stages "lorenzo->diff->bitshuffle->rze" -e 1e-4
```

## Decompress, compare, and report

```bash
fzgmod-cli -x -i compressed.fzm -o decompressed.f32 --compare data.f32 --report
# Prints: Output size, Time, Throughput, Value Range, Max Abs Error, PSNR, NRMSE
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
