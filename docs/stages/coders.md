# Coder stages {#stage_coders}

| Stage | Description |
|---|---|
| \subpage stage_huffman | GPU Huffman entropy coding (cuSZ port) |
| \subpage stage_ans | GPU rANS entropy coding (dietGPU, byte-level) |
| \subpage stage_gpulz | GPU LZSS (LZ77 + flag-bitmap) lossless compression (GPULZ) |
| \subpage stage_rle | Run-length encoding |
| \subpage stage_rze | Zero-word bitmap reducer with recursive bitmap compression (LC component) |
| \subpage stage_rre | Repeated-word bitmap reducer with recursive bitmap compression (LC component) |
| \subpage stage_rare | Adaptive top-bit matching generalization of RRE (LC component) |
| \subpage stage_raze | Adaptive leading-zero-bit generalization of RZE (LC component) |
| \subpage stage_clog | Per-subchunk leading-zero compression and adaptive bit packing (LC framework lossless component) |
| \subpage stage_hclog | CLOG bit packing with per-subchunk TCMS selection (LC framework lossless component) |
| \subpage stage_bitpack | Dense bit-packing of fixed-width integers |
| \subpage stage_adaptive_bitpack | Per-block adaptive fixed-rate bit-plane coding (cuSZp plain mode) |
| \subpage stage_speck2d | GPU-parallel "wavefront" SPECK-like coder (2-D), decode-parallel format |
| \subpage stage_outlier_correct | Sparse exact outlier correction — turns a coefficient-domain quantization bound into a guaranteed reconstructed-domain pointwise bound, transform-agnostic |
