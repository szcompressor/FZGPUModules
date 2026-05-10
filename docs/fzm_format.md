# FZM File Format {#fzm_format}

`.fzm` files store a compressed payload together with everything needed to
decompress it: the full pipeline stage graph, per-stage configuration, and buffer
layout.  You can decompress an `.fzm` file using only this specification and the
element data types — no pipeline source code required.

All multi-byte integers are **little-endian**.
The library enforces a little-endian host at configure time.

---

## Version History

| Version | `FZMHeaderCore` size | Changes |
|---------|----------------------|---------|
| v3.0    | 72 bytes             | Initial versioned format |
| v3.1    | 80 bytes             | Added `flags`, `data_checksum`, `header_checksum` fields |

**Version field (2 bytes):** `high_byte = major`, `low_byte = minor`.
A major mismatch causes the read to throw.
A minor mismatch emits a warning and continues.
Legacy files stored a plain integer (e.g. `3`); these are read as major=3, minor=0.

---

## File Layout

```
┌───────────────────────────────────────────────────┐
│  FZMHeaderCore                     (80 bytes)     │
├───────────────────────────────────────────────────┤
│  FZMStageInfo × num_stages                        │
│  (256 bytes each)                                 │
├───────────────────────────────────────────────────┤
│  FZMBufferEntry × num_buffers                     │
│  (256 bytes each)                                 │
├───────────────────────────────────────────────────┤
│  Compressed payload                               │
│  Starts at byte offset  header_size               │
│                                                   │
│    segment 0   (FZMBufferEntry[0].data_size bytes)│
│    segment 1   (FZMBufferEntry[1].data_size bytes)│
│    ...                                            │
└───────────────────────────────────────────────────┘
```

The payload is a flat concatenation of buffer segments, one per `FZMBufferEntry`.
Each entry's `byte_offset` field gives the segment's start position **relative to
`header_size`** (i.e. relative to the start of the payload, not the start of the file).

---

## FZMHeaderCore (80 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0  | 4 | `magic`            | Must equal `0x464D5A32` ("FZM2" LE) |
| 4  | 2 | `version`          | `(major << 8) \| minor`; current = `0x0301` |
| 6  | 2 | `num_buffers`      | Number of `FZMBufferEntry` records |
| 8  | 8 | `uncompressed_size`| Total uncompressed input size (bytes) |
| 16 | 8 | `compressed_size`  | Total compressed payload size (bytes) |
| 24 | 8 | `header_size`      | Byte offset where the compressed payload begins |
| 32 | 4 | `num_stages`       | Number of `FZMStageInfo` records |
| 36 | 2 | `num_sources`      | Number of pipeline source stages (currently always 1) |
| 38 | 2 | `flags`            | Feature flags (see below) |
| 40 | 32| `source_uncompressed_sizes[4]` | Per-source uncompressed size (8 bytes × 4 slots); only index 0 is used |
| 72 | 4 | `data_checksum`    | CRC32 (IEEE 802.3) of payload; 0 if flag not set |
| 76 | 4 | `header_checksum`  | CRC32 of full header with this field zeroed; 0 if flag not set |

**Flags:**

| Bit | Constant | Meaning |
|-----|----------|---------|
| 0   | `FZM_FLAG_HAS_DATA_CHECKSUM`   | `data_checksum` is valid |
| 1   | `FZM_FLAG_HAS_HEADER_CHECKSUM` | `header_checksum` is valid |

**Header checksum computation:**
1. Concatenate the core (80 B) + stage array + buffer array.
2. Zero the 4 bytes at offset 76 (`header_checksum` field).
3. CRC32 the entire buffer.

---

## FZMStageInfo (256 bytes, one per stage)

Describes one stage in the pipeline: its type, serialized configuration, and which
DAG buffers feed its inputs and outputs.

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0   | 2   | `stage_type`        | `StageType` enum value |
| 2   | 2   | `stage_version`     | Stage config format version |
| 4   | 1   | `num_inputs`        | Number of input ports |
| 5   | 1   | `num_outputs`       | Number of output ports |
| 6   | 2   | `reserved1`         | Reserved (zeroed) |
| 8   | 16  | `input_buffer_ids[8]`  | DAG buffer ID for each input (2 bytes each); `0xFFFF` = unused |
| 24  | 16  | `output_buffer_ids[8]` | DAG buffer ID for each output (2 bytes each); `0xFFFF` = unused |
| 40  | 128 | `stage_config`      | Stage-defined serialized config (see `Stage::serializeHeader()`) |
| 168 | 4   | `config_size`       | Valid bytes in `stage_config` |
| 172 | 84  | *(reserved)*        | Reserved (zeroed) |

---

## FZMBufferEntry (256 bytes, one per buffer)

Describes one compressed buffer segment: its producer, element type, sizes, and
position within the payload.

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0   | 2  | `stage_type`          | Producer stage type |
| 2   | 2  | `stage_version`       | Producer stage config version |
| 4   | 1  | `data_type`           | `DataType` enum value |
| 5   | 1  | `producer_output_idx` | Which output port of the producer stage |
| 6   | 2  | `dag_buffer_id`       | DAG routing ID; `0xFFFF` = unassigned |
| 8   | 64 | `name[64]`            | Output port name, null-terminated |
| 72  | 8  | `data_size`           | Compressed bytes in this segment |
| 80  | 8  | `allocated_size`      | Buffer capacity needed for decompression |
| 88  | 8  | `uncompressed_size`   | Bytes after fully decompressing this stage's output |
| 96  | 8  | `byte_offset`         | Offset of this segment within the payload (relative to `header_size`) |
| 104 | 128| `stage_config`        | Producer stage config (same content as the matching `FZMStageInfo.stage_config`) |
| 232 | 4  | `config_size`         | Valid bytes in `stage_config` |
| 236 | 20 | *(reserved)*          | Reserved (14 bytes declared + 6 bytes implicit struct padding) |

---

## StageType Values

| Value | Constant | Stage |
|-------|----------|-------|
| 0  | `UNKNOWN`      | Unknown / unset |
| 1  | `LORENZO_QUANT`| `LorenzoQuantStage` — fused float predictor + quantizer; dimensionality stored in config |
| 2  | `DIFFERENCE`   | `DifferenceStage` — first-order difference coder |
| 3  | `SCALE`        | `ScaleStage` (test utility) |
| 4  | `PASSTHROUGH`  | `PassThroughStage` (test utility) |
| 5  | `RLE`          | `RLEStage` — run-length encoding |
| 6  | `HUFFMAN`      | Reserved for future use |
| 7  | `BITPACK`      | `BitpackStage` — dense N-bit integer packing |
| 10 | `SPLIT`        | `SplitStage` (test utility) |
| 11 | `MERGE`        | `MergeStage` (test utility) |
| 12 | `LORENZO`      | `LorenzoStage` — plain integer delta predictor; dimensionality stored in config |
| 14 | `QUANTIZER`    | `QuantizerStage` — direct-value quantizer |
| 15 | `ZIGZAG`       | `ZigzagStage` — zigzag encode/decode |
| 16 | `NEGABINARY`   | `NegabinaryStage` — negabinary encode/decode |
| 17 | `BITSHUFFLE`   | `BitshuffleStage` — GPU bit-matrix transpose |
| 18 | `RZE`          | `RZEStage` — recursive zero-byte elimination |

**Rule:** never reuse or renumber an existing value — stage type IDs are baked into
`.fzm` files on disk.  New stages always take the next unused integer.

---

## DataType Values

| Value | Constant  | C type     |
|-------|-----------|------------|
| 0     | `UINT8`   | `uint8_t`  |
| 1     | `UINT16`  | `uint16_t` |
| 2     | `UINT32`  | `uint32_t` |
| 3     | `UINT64`  | `uint64_t` |
| 4     | `INT8`    | `int8_t`   |
| 5     | `INT16`   | `int16_t`  |
| 6     | `INT32`   | `int32_t`  |
| 7     | `INT64`   | `int64_t`  |
| 8     | `FLOAT32` | `float`    |
| 9     | `FLOAT64` | `double`   |
| 255   | `UNKNOWN` | byte-transparent (type checking skipped at pipeline finalize) |

---

## Reading a File Without the Library

Minimum steps to parse an `.fzm` file in Python:

```python
import struct, zlib, sys

STAGE_TYPES = {
    0: "Unknown", 1: "LorenzoQuant", 2: "Difference", 3: "Scale",
    4: "PassThrough", 5: "RLE", 6: "Huffman", 7: "BitPack",
    10: "Split", 11: "Merge", 12: "Lorenzo", 14: "Quantizer",
    15: "Zigzag", 16: "Negabinary", 17: "Bitshuffle", 18: "RZE",
}
DATA_TYPES = {
    0: "uint8", 1: "uint16", 2: "uint32", 3: "uint64",
    4: "int8", 5: "int16", 6: "int32", 7: "int64",
    8: "float32", 9: "float64", 0xFF: "unknown",
}

filename = sys.argv[1] if len(sys.argv) > 1 else "output.fzm"
print(f"Parsing: {filename}\n")

with open(filename, "rb") as f:
    # 1. Read and validate the header core
    core = f.read(80)
    magic, version, num_buffers = struct.unpack_from("<IHH", core, 0)
    assert magic == 0x464D5A32, f"Bad magic: 0x{magic:08X}"
    major = (version >> 8) if version > 0xFF else version
    minor = (version & 0xFF) if version > 0xFF else 0
    assert major == 3, f"Unsupported major version {major}"

    uncomp_size, comp_size, header_size = struct.unpack_from("<QQQ", core, 8)
    num_stages, num_sources, flags      = struct.unpack_from("<IHH", core, 32)
    data_crc, hdr_crc = struct.unpack_from("<II", core, 72)

    print(f"=== Header ===")
    print(f"  magic:            0x{magic:08X} (FZM2)")
    print(f"  version:          {major}.{minor}")
    print(f"  num_stages:       {num_stages}")
    print(f"  num_buffers:      {num_buffers}")
    print(f"  num_sources:      {num_sources}")
    print(f"  uncompressed:     {uncomp_size:,} bytes")
    print(f"  compressed:       {comp_size:,} bytes")
    print(f"  header_size:      {header_size:,} bytes")
    print(f"  flags:            0x{flags:04X}  (data_crc={'yes' if flags&1 else 'no'}, hdr_crc={'yes' if flags&2 else 'no'})")
    if uncomp_size:
        print(f"  compression ratio: {uncomp_size/comp_size:.3f}x")

    # 2. Read stage and buffer arrays
    stage_array  = f.read(num_stages  * 256)
    buffer_array = f.read(num_buffers * 256)

    # 3. Verify header checksum (v3.1+)
    if flags & 0x0002:
        full_header = bytearray(core) + stage_array + buffer_array
        full_header[76:80] = b'\x00\x00\x00\x00'  # zero header_checksum before computing
        computed = zlib.crc32(full_header) & 0xFFFFFFFF
        assert computed == hdr_crc, f"Header CRC mismatch: computed 0x{computed:08X}, stored 0x{hdr_crc:08X}"
        print(f"\n  header CRC:  PASS (0x{hdr_crc:08X})")
    else:
        print(f"\n  header CRC:  skipped (flag not set)")

    # 4. Read the compressed payload
    f.seek(header_size)
    payload = f.read(comp_size)

    # 5. Verify data checksum (v3.1+)
    if flags & 0x0001:
        computed = zlib.crc32(payload) & 0xFFFFFFFF
        assert computed == data_crc, f"Payload CRC mismatch: computed 0x{computed:08X}, stored 0x{data_crc:08X}"
        print(f"  payload CRC: PASS (0x{data_crc:08X})")
    else:
        print(f"  payload CRC: skipped (flag not set)")

    # 6. Decode stage info
    print(f"\n=== Stages ({num_stages}) ===")
    for i in range(num_stages):
        entry = stage_array[i*256 : (i+1)*256]
        stage_type_id, stage_ver, num_in, num_out = struct.unpack_from("<HHBB", entry, 0)
        # input/output buffer IDs: 8 x uint16 each, starting at offset 8 and 24
        in_ids  = [struct.unpack_from("<H", entry, 8  + j*2)[0] for j in range(num_in)]
        out_ids = [struct.unpack_from("<H", entry, 24 + j*2)[0] for j in range(num_out)]
        in_str  = ", ".join(str(x) for x in in_ids)  if in_ids  else "-"
        out_str = ", ".join(str(x) for x in out_ids) if out_ids else "-"
        sname = STAGE_TYPES.get(stage_type_id, f"type#{stage_type_id}")
        print(f"  stage[{i}]: {sname} v{stage_ver}  inputs=[{in_str}]  outputs=[{out_str}]")

    # 7. Decode buffer entries and extract segments
    print(f"\n=== Buffers ({num_buffers}) ===")
    segments = []
    for i in range(num_buffers):
        entry = buffer_array[i*256 : (i+1)*256]
        stage_type_id, stage_ver  = struct.unpack_from("<HH", entry, 0)
        data_type_id, prod_out_idx = struct.unpack_from("<BB", entry, 4)
        dag_buf_id                 = struct.unpack_from("<H",  entry, 6)[0]
        name = entry[8:72].rstrip(b'\x00').decode("utf-8", errors="replace")

        data_size        = struct.unpack_from("<Q", entry, 72)[0]
        allocated_size   = struct.unpack_from("<Q", entry, 80)[0]
        uncompressed_size= struct.unpack_from("<Q", entry, 88)[0]
        byte_offset      = struct.unpack_from("<Q", entry, 96)[0]

        sname = STAGE_TYPES.get(stage_type_id, f"type#{stage_type_id}")
        dtype = DATA_TYPES.get(data_type_id, f"dtype#{data_type_id}")

        print(f"  buf[{i}]: '{name}'  stage={sname}  dtype={dtype}  "
              f"data_size={data_size:,}  uncomp={uncompressed_size:,}  offset={byte_offset:,}")

        segment = payload[byte_offset : byte_offset + data_size]
        if len(segment) != data_size:
            print(f"    WARNING: expected {data_size} bytes but got {len(segment)}")
        segments.append(segment)

    print(f"\nPayload size: {len(payload):,} bytes")
    print(f"Total segment bytes: {sum(len(s) for s in segments):,}")
    print(f"\nParsed OK.")
```

This produces output like the following example:

```txt
Parsing: output.fzm

=== Header ===
  magic:            0x464D5A32 (FZM2)
  version:          3.1
  num_stages:       2
  num_buffers:      4
  num_sources:      1
  uncompressed:     262,144 bytes
  compressed:       250,280 bytes
  header_size:      1,616 bytes
  flags:            0x0003  (data_crc=yes, hdr_crc=yes)
  compression ratio: 1.047x

  header CRC:  PASS (0xF02D1EA0)
  payload CRC: PASS (0x5ECF8D8B)

=== Stages (2) ===
  stage[0]: LorenzoQuant v1  inputs=[5]  outputs=[0, 1, 2, 3]
  stage[1]: RLE v1  inputs=[0]  outputs=[4]

=== Buffers (4) ===
  buf[0]: 'output'  stage=RLE  dtype=uint16  data_size=250,276  uncomp=250,276  offset=0
  buf[1]: 'outlier_errors'  stage=LorenzoQuant  dtype=float32  data_size=0  uncomp=0  offset=250,276
  buf[2]: 'outlier_indices'  stage=LorenzoQuant  dtype=uint32  data_size=0  uncomp=0  offset=250,276
  buf[3]: 'outlier_count'  stage=LorenzoQuant  dtype=uint32  data_size=4  uncomp=4  offset=250,276

Payload size: 250,280 bytes
Total segment bytes: 250,280

Parsed OK.
```

---

## Versioning Rules

- **Major bump** — incompatible layout change to any header struct.  Old files are rejected at read time.
- **Minor bump** — additive change only (new fields in reserved space, new flag bits).  Old files load with new fields defaulting to 0; new files load on old readers with a warning.
- **stage_version** — per-stage config version managed by each stage independently.  The library does not interpret it; stages read it in `Stage::deserializeHeader()` to handle their own config migration.

| Library version | FZM versions supported |
|----------------|------------------------|
| `v1.x`         | v3.0, v3.1             |
| `v2.x` (current) | v3.1 (reads v3.0 with warning) |
