# GPULZStage {#stage_gpulz}

**Header:** `modules/coders/gpulz/gpulz_stage.h`
**Class:** `fz::GPULZStage` — no template parameters
**Category:** Coder (lossless)

**Common instantiation:**
```cpp
auto* gpulz = p.addStage<fz::GPULZStage>();
gpulz->setChunkSize(2048);  // bytes; 1024, 2048, or 4096 (default 2048)
gpulz->setWordSize(4);      // 1, 2, 4, or 8 (default 4)
```

---

## What it does

GPU LZSS (LZ77 + a literal/match flag bitmap) — a direct port of the
**GPULZ** reference kernels. Each fixed-size chunk (`chunk_size` bytes) is
compressed independently by one CUDA thread block:

- The chunk is loaded into shared memory as `word_size`-byte elements.
- Every element finds the **longest** repeated run within a 32-element
  sliding window behind it, via an offset-indexed equality mask (see
  [Match search](#match-search) below).
- A single thread then serially walks the per-element match lengths to build
  a literal/match decision for each "encode item" and a 1-bit-per-item flag
  bitmap (this walk is inherently sequential, matching the upstream
  algorithm, and is now the largest single cost in the encode kernel).
- A block-wide `cub::BlockScan` turns per-item byte sizes into packing
  offsets; items are emitted into a shared staging buffer and copied out in
  coalesced 32-bit stores as literal bytes (`word_size` bytes) or match
  tokens (`[length, offset]`, 2 bytes).

This is the first LZ77-family stage in FZGPUModules — pairing it with
`HuffmanStage` or `ANSStage` reproduces the LZ77 + entropy-coding structure of
DEFLATE/Zstandard, e.g. `GPULZ -> Huffman` or `GPULZ -> ANS`.

---

## Stage settings

```cpp
gpulz->setChunkSize(2048);  // bytes: 1024, 2048, or 4096 (default 2048)
gpulz->setWordSize(4);      // word granularity in bytes: 1, 2, 4, or 8 (default 4)
gpulz->setMatchLevel(1);    // match-search effort: 0 or 1 (default 1)
```

`match_level` is an **encode-side** knob only — it changes how hard the encoder
looks for matches, not what the stream looks like. It is not serialized, and a
stream produced at either level decodes identically.

| level | search | enc GB/s | ratio |
|---|---|---|---|
| 0 | exact longest match over the 32-element near window | 124 | 4.36x |
| 1 | + hashed long-range candidates (offsets to 255) | 109 | 5.08x |

(H100, 24.7 MB of Lorenzo-quantized `CLDHGH` residuals, `chunk_size=2048`,
`word_size=4`. At `chunk_size=4096` level 1 reaches 5.30x.)

`chunk_size` is restricted to this set because the encode kernel keeps the
whole chunk (plus per-element length/offset/prefix-sum scratch) in **static**
`__shared__` memory, and the algorithm's Blelloch scan requires
`chunk_size / word_size` to be a power of two `>= 128` (the fixed
128-thread block size) — all three supported sizes satisfy this for every
supported `word_size`.

The 32-element sliding window and the 128-thread block size are fixed,
matching the upstream reference's defaults; they are not currently exposed as
stage settings.

---

## Match search {#match-search}

Upstream GPULZ advances its window pointer on every step of the search, so a
match it finds consumes the candidate offsets behind it — it is a greedy
approximation that systematically settles for a shorter match than exists.
This stage instead performs an **exact** longest-match search: for each
element it builds an offset-indexed equality mask (bit `o-1` of `omask[i]` is
set iff `buffer[i] == buffer[i-o]`), then AND-s successive masks. Each AND
drops exactly the offsets that just stopped matching, so the extension loop
runs `longest_match + 1` times for *all* candidate offsets at once rather
than once per offset. It is both substantially cheaper and strictly better
compressing.

One consequence: a match may now be **longer than the 32-element window**
(e.g. a long constant run reached at offset 1). Lengths are capped at 255,
the limit of the token's one-byte length field.

### Hashed long-range matcher (`match_level` 1)

The near window is exact but costs one comparison per element per offset, so
it cannot be widened far — while measurement showed compression ratio still
climbing steeply with window size (4.36x at 32, 4.82x at 48, 5.13x at 64).
A hash lookup instead finds a candidate in O(1) no matter how far back it is.

The chunk is walked in 8 sub-blocks; the table only ever holds positions from
*earlier* sub-blocks, so a hit is always a legal back-reference and the lookup
and insert for a sub-block are simply separated by a barrier. Two details
matter:

- The key covers **two** consecutive words. On quantized data most single
  values are small integers that recur constantly, so a one-word table mostly
  rediscovers a nearby duplicate the near window already covers; requiring two
  words makes a hit imply a match worth extending (4.70x → 5.01x at identical
  throughput).
- Positions where the near window already found a long match skip the lookup
  (130 vs 117 GB/s, unchanged ratio).

Buckets are filled with `atomicMax` on position+1 rather than a plain store.
Any bucket occupant would be a legal candidate, so a racing store would still
be *correct* — but the winner, and therefore the compressed bytes, would vary
run to run. The atomic keeps the most recent position, which is both the
nearest candidate and a deterministic one.

---

## Parallel decode

Upstream's `decompressKernel` decodes a whole chunk on a single thread,
replaying the item stream in order. That serial dependency is only apparent,
and this stage decodes a chunk with a whole CUDA block instead:

1. The flag bitmap alone fixes every item's **input** size (2 bytes for a
   match token, `word_size` for a literal), so one block scan recovers every
   item's input offset.
2. Reading each match's length byte at its now-known input offset gives every
   item's **output** length; a second block scan recovers every item's output
   position.
3. The only genuinely sequential structure — back-references chaining onto
   other back-references — is resolved by **pointer doubling** over a
   per-element source-index array (a match element `q` sources from
   `q - offset`, always `< q`, so chains terminate), for
   `log2(chunk elements)` parallel rounds with early exit.
4. A final gather materializes the output from the literal values.

At `chunk_size=2048, word_size=4` this is ~2.4x the throughput of the
one-thread-per-chunk decode; at `chunk_size=4096, word_size=4`, ~4x.
`word_size=1` with `chunk_size=4096` is the weak configuration (4096 elements
per chunk, ~24 KB of shared memory per block) and does not benefit.

---

## Split mode (Zstandard-style literals/sequences separation) {#split-mode}

```cpp
gpulz->setSplitMode(true);   // default false
```

Emits four output ports instead of one interleaved stream:

| port | contents | suggested coder |
|---|---|---|
| `literals` | literal words, back to back (raw-fallback chunks land here too) | `HuffmanStage<uint16_t>` |
| `lengths`  | one match-length byte per match token | `ANSStage` |
| `offsets`  | one match-offset byte per match token | `ANSStage` |
| `meta`     | stream header + per-chunk size table + flag bitmaps | `ANSStage` |

This is the split Zstandard makes for the same reason: the parts have very
different symbol distributions, and interleaving them raises the entropy any
one coder sees. Measured across six SDRB fields, coding the four ports
separately beats the single-stream form by **23-43% compression ratio**.

Roughly half that gain comes not from the split itself but from the
`literals` port keeping the data's **natural word alphabet**. Lorenzo-quantized
uint16 codes carry strong correlation between a code's high and low byte;
coding them as bytes throws it away (4.14 bits/byte vs 3.43 when coded as
uint16 symbols). Feed `literals` to a symbol-width-matched coder.

### Everything must be coded, and everything must be merged

The single-stream form entropy codes the whole payload by construction. A split
leaks any byte left out of a port, so two categories are deliberately folded in
rather than parked in an uncoded tail:

- **raw-fallback chunks** go into `literals` (such a chunk is all literal by
  definition). Leaving them out cost 28% CR on EXAALT, where nearly every chunk
  is raw fallback.
- **the per-chunk size table** goes into `meta`. It scales with chunk count and
  is highly repetitive; left raw it cost 67% CR on AEROD_v, whose 50 KB size
  table alone exceeded the entire single-stream archive.

Any size gate in front of a coder should skip only the *launch* for a stream too
small to pay for it, then keep `min(raw, coded)`. A gate that *forces* raw is a
trap: an early 64 KB threshold made AEROD_v 4x worse, because all four of its
streams sat under it.

### GPU-specific divergences from the CPU format

- Zstd interleaves the three sequence streams into one bitstream with three FSE
  states, because on a CPU that keeps all three states in registers through a
  single sequential decode loop. On a GPU that inverts: same-level DAG stages
  run concurrently on separate streams, so separate ports let their coders
  overlap.
- Zstd encodes lengths/offsets as a small *code* plus raw *extra bits*, keeping
  the FSE alphabet tiny while reaching values past 65536. Our length and offset
  fields are one byte each, so the alphabet is already <=256 and the split buys
  nothing. It becomes necessary only if the match window is widened past 255.

### Decode path

Split mode does **not** rebuild the packed single-stream form to decode. A
dedicated kernel decodes straight out of the four ports: the flag bitmap fixes
every item's input size, so one block scan recovers item offsets and two more
give each item its slot in the literals and token streams, which are then read
at those indices. Restriping first would run those same scans twice and move the
whole uncompressed-size intermediate through global memory both ways — measured
7.5x slower for the inverse stage (3.59 ms vs 0.476 ms on 61 MB).

With that in place the pipeline's decompress cost is dominated by the literals
**Huffman**, not by this stage.

### Throughput notes

`sublen` (elements per Huffman coarse-encode partition) is hardcoded to 768 in
the vendored PHF code, and the coarse decode runs one thread per partition — so
a 2.68 M-symbol literals stream decodes on only ~3500 threads. Dropping it to
256 measured ~1.4x faster decompress for -2.3% CR on QCLOUD. It is a genuine
CR/throughput lever but is not currently exposed as a stage setting, and
changing the constant affects every `HuffmanStage` user.

See `examples/gpu_zstd.cpp` for the full pipeline.

---

## Alignment requirement

None. `getRequiredInputAlignment()` returns 1 and the stage zero-pads its own
tail chunk.

Requesting pipeline-level alignment does not work for a coder behind a
width-changing stage: `Pipeline::finalize()` pads the *pipeline input* to the
LCM of all stage alignments, but `LorenzoQuantStage` turns float32 into uint16
codes, so a 2048-aligned input arrives here as half as many bytes and need not
be aligned at all. Forcing the input up additionally grows the upstream stage's
output past its own estimate and trips the buffer-overwrite check.

---

## Raw-chunk fallback

If a chunk's LZSS-encoded form (flag bitmap + literal/match bytes) would not
be smaller than the chunk itself, the chunk is stored verbatim instead (a
high bit on its header entry marks this). This bounds worst-case output to
`original_size + header_bytes` regardless of input entropy.

---

## All-zero-chunk fast path

Before running the match search, the encode kernel does a warp-vote check
(`fz::backend::anySync32`) for whether the whole chunk is zero. If so, the
match search and flag/data encode are skipped entirely — the chunk
contributes 0 bytes to the compressed payload (a `(flag_size=0, data_size=0)`
sentinel, distinct from the raw-fallback sentinel above). On decode, the
corresponding output span is zero-filled directly rather than walking a flag
bitmap. This matters for sparse inputs (e.g. quantized neural-compressor
latents, which are often mostly zero) where whole 1-4 KB chunks are
frequently all-zero — those chunks cost nothing beyond the 8-byte header
entry, on both the encode and decode side.

---

## Graph capture

Forward (compress) is CUDA-graph capturable — the final output-size readback
is deferred to `postStreamSync()`. The inverse (decompress) path is not — it
reads the stream header (original size, per-chunk flag/data sizes) with
blocking device-to-host copies before it can compute per-chunk input offsets
and launch the decode kernel. This mirrors `RREStage`/`RZEStage`.

---

## Output stream format

```
[uint32_t: original byte count]
[uint32_t: num_chunks]
[ (uint32_t flag_size, uint32_t data_size) x num_chunks ]   // flag_size high bit -> chunk stored raw
[ per-chunk payload: flag bytes, then compressed-data bytes (or raw bytes if flagged) ... ]
```

---

## Acknowledgements

The per-chunk stream format and the sequential literal/match parse in
`GPULZStage` follow `compressKernelI` from **GPULZ**. The match search, the
prefix sum, the staged data writes and the parallel decode are FZGM's own.

> Boyuan Zhang, Jiannan Tian, Sheng Di, Xiaodong Yu, Martin Swany, Dingwen Tao,
> and Franck Cappello.
> *GPULZ: Optimizing LZSS Lossless Compression for Multi-byte Data on Modern GPUs.*
> ICS '23. https://github.com/hpdps-group/ICS23-GPULZ

The all-zero-chunk fast path is adapted from the "sparse" GPULZ variant in
**AIZ_VLDB26** (Boyuan Zhang, `test/gpulz.cuh`'s `notEmptyFlagArr`):
https://github.com/boyuanzhang62/AIZ_VLDB26

See `THIRD_PARTY.md` — neither upstream repository declares an explicit
license.
