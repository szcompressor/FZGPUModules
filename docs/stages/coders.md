# Coder stages {#stage_coders}

| Stage | Description |
|---|---|
| \subpage stage_huffman | GPU Huffman entropy coding (PHF coarse-grained) |
| \subpage stage_ans | GPU rANS entropy coding (dietGPU, byte-level) |
| \subpage stage_rle | Run-length encoding |
| \subpage stage_rze | Recursive zero-byte elimination |
| \subpage stage_rre | Repetition-reduction encoding (LC framework lossless component) |
| \subpage stage_rare | Repetition-adaptive reduction encoding (LC framework, auto-k generalization of RRE) |
| \subpage stage_raze | Zero-adaptive reduction encoding (LC framework, auto-k generalization of RZE) |
| \subpage stage_clog | Compressed-Logarithm adaptive bit-width coding (LC framework lossless component) |
| \subpage stage_hclog | Compressed-Logarithm coding with per-subchunk TCMS fallback (LC framework lossless component) |
| \subpage stage_bitpack | Dense bit-packing of fixed-width integers |
| \subpage stage_adaptive_bitpack | Per-block adaptive fixed-rate bit-plane coding (cuSZp plain mode) |
