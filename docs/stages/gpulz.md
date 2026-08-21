# GPULZStage {#stage_gpulz}

**Header:** `modules/coders/gpulz/gpulz_stage.h`
**Class:** `fz::GPULZStage`
**Category:** Coder (lossless)

```cpp
auto* gpulz = p.addStage<fz::GPULZStage>();
gpulz->setChunkSize(2048);
gpulz->setWordSize(4);
gpulz->setMatchLevel(1);
```

---

## What it does

`GPULZStage` is a chunk-parallel GPU LZSS coder derived from **GPULZ**. It
replaces repeated word sequences with one-byte length/offset pairs and stores
unmatched words as literals. Chunks are independent, so incompressible chunks
can be stored raw and decoded without affecting their neighbors.

The current kernels are a substantial FZGM rewrite. They retain GPULZ's
per-chunk stream grammar and sequential literal/match parse, but use FZGM match
search, packing, container handling, and parallel decoder.

---

## Settings

| Setting | Default | Allowed values | Effect |
|---|---:|---|---|
| `chunk_size` | `2048` | `1024`, `2048`, `4096` | Independent chunk size in bytes |
| `word_size` | `4` | `1`, `2`, `4`, `8` | Literal and match granularity in bytes |
| `match_level` | `1` | `0`, `1` | `0`: 32-word near-window search; `1`: also try hashed long-range candidates |
| `split_mode` | `false` | Boolean | Emit separate literal, length, offset, and metadata ports |

`match_level` affects compression only; both levels produce the same stream
format. Matches are limited to a length and backward offset of 255 words.

Equivalent TOML:

```toml
[[stage]]
name = "lz"
type = "GPULZ"
chunk_size = 2048
word_size = 4
match_level = 1
split_mode = false
```

---

## Ports

Normal mode has one byte-stream output named `output`.

With `setSplitMode(true)`, the stage instead emits:

| Port | Contents |
|---|---|
| `literals` | Literal words, including raw-fallback chunks |
| `lengths` | One byte per match length |
| `offsets` | One byte per match offset |
| `meta` | Header, chunk sizes, and flag bitmaps |

The split ports can be coded independently; use a coder whose symbol width
matches `word_size` for `literals`. The inverse stage consumes the four ports
directly rather than rebuilding the interleaved stream. See
`examples/gpu_zstd.cpp` for a complete pipeline.

---

## Constraints and behavior

- Input is an arbitrary byte stream; no pipeline-level alignment is required.
  A partial final chunk is padded internally and restored to its original size.
- A chunk whose encoded representation would be no smaller is stored raw. This
  bounds expansion to the input size plus container metadata.
- All-zero chunks use an empty-payload sentinel and are reconstructed by
  zero-filling their output span.
- The forward path supports CUDA Graph capture because output-size readback is
  deferred until `postStreamSync()`. The inverse path is not graph-compatible;
  it reads the stream header before computing chunk offsets and launching decode.

---

## Normal output format

```text
[uint32_t original_byte_count]
[uint32_t num_chunks]
[(uint32_t flag_size, uint32_t data_size) x num_chunks]
[chunk payloads: flag bytes followed by encoded or raw data]
```

The high bit of `flag_size` marks a raw chunk. A zero `flag_size` and
`data_size` marks an all-zero chunk.

---

## Acknowledgements

The per-chunk format and sequential literal/match parse follow
`compressKernelI` from **GPULZ**:

> Boyuan Zhang, Jiannan Tian, Sheng Di, Xiaodong Yu, Martin Swany, Dingwen Tao,
> and Franck Cappello. *GPULZ: Optimizing LZSS Lossless Compression for
> Multi-byte Data on Modern GPUs.* ICS '23.
> https://github.com/hpdps-group/ICS23-GPULZ

The all-zero fast path is adapted from the sparse GPULZ variant in
**AIZ_VLDB26**: https://github.com/boyuanzhang62/AIZ_VLDB26

Neither upstream repository declares an explicit license. See
`THIRD_PARTY.md` for the provenance and redistribution notice.
