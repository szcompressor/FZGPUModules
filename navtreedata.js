/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "FZGPUModules", "index.html", [
    [ "Overview", "index.html#autotoc_md8", [
      [ "Requirements", "index.html#autotoc_md10", null ]
    ] ],
    [ "Quick Start", "index.html#autotoc_md12", [
      [ "Building from Source", "index.html#autotoc_md13", null ],
      [ "C++ API Usage", "index.html#autotoc_md14", null ],
      [ "Available Stages", "index.html#mainpage_stages", null ],
      [ "Memory Strategies", "index.html#autotoc_md16", null ],
      [ "Caller-Allocated Output", "index.html#autotoc_md18", null ],
      [ "CUDA Graph Support", "index.html#autotoc_md20", null ],
      [ "Compressor Config File", "index.html#autotoc_md22", null ],
      [ "File I/O", "index.html#autotoc_md24", null ],
      [ "Decode-only pipelines (no warmup compress)", "index.html#autotoc_md26", null ],
      [ "Thread Safety", "index.html#autotoc_md28", null ]
    ] ],
    [ "Citation", "index.html#autotoc_md30", null ],
    [ "Building from Source", "building_from_source.html", [
      [ "Prerequisites", "building_from_source.html#autotoc_md32", null ],
      [ "Getting the Source", "building_from_source.html#autotoc_md34", null ],
      [ "Quick Build", "building_from_source.html#autotoc_md36", null ],
      [ "CMake Presets", "building_from_source.html#autotoc_md38", null ],
      [ "CMake Options", "building_from_source.html#autotoc_md40", null ],
      [ "Common Build Examples", "building_from_source.html#autotoc_md42", null ],
      [ "Binary Output", "building_from_source.html#autotoc_md44", null ],
      [ "Testing", "building_from_source.html#autotoc_md46", [
        [ "Host Sanitizers (ASan + UBSan)", "building_from_source.html#autotoc_md47", null ],
        [ "CUDA Compute Sanitizer", "building_from_source.html#autotoc_md48", null ],
        [ "ThreadSanitizer", "building_from_source.html#autotoc_md49", null ]
      ] ],
      [ "Install", "building_from_source.html#autotoc_md51", null ],
      [ "Using from CMake", "building_from_source.html#autotoc_md53", null ],
      [ "Generating Documentation", "building_from_source.html#autotoc_md55", null ]
    ] ],
    [ "Stage Reference", "stages_overview.html", [
      [ "Categories", "stages_overview.html#autotoc_md210", null ],
      [ "Fused stages", "stage_fused.html", [
        [ "LorenzoQuantStage", "stage_lorenzo_quant.html", [
          [ "What it does", "stage_lorenzo_quant.html#autotoc_md225", null ],
          [ "Template parameters", "stage_lorenzo_quant.html#autotoc_md227", null ],
          [ "Available instantiations", "stage_lorenzo_quant.html#autotoc_md228", null ],
          [ "Stage settings", "stage_lorenzo_quant.html#autotoc_md230", null ],
          [ "Output ports (compression)", "stage_lorenzo_quant.html#autotoc_md232", null ],
          [ "Error bound modes", "stage_lorenzo_quant.html#autotoc_md234", null ],
          [ "Dimension setup — critical ordering rule", "stage_lorenzo_quant.html#autotoc_md236", null ],
          [ "Value base and CUDA Graph capture", "stage_lorenzo_quant.html#autotoc_md238", null ],
          [ "Typical pipeline", "stage_lorenzo_quant.html#autotoc_md240", null ],
          [ "Acknowledgements", "stage_lorenzo_quant.html#autotoc_md242", null ]
        ] ],
        [ "GInterpStage", "stage_ginterp.html", [
          [ "What it does", "stage_ginterp.html#autotoc_md163", [
            [ "Why this stage is fused (no standalone predictor)", "stage_ginterp.html#autotoc_md164", null ]
          ] ],
          [ "Template parameters", "stage_ginterp.html#autotoc_md166", null ],
          [ "Available instantiations", "stage_ginterp.html#autotoc_md167", [
            [ "Precision and shared memory", "stage_ginterp.html#autotoc_md168", null ]
          ] ],
          [ "Stage settings", "stage_ginterp.html#autotoc_md170", [
            [ "Radius auto-tune", "stage_ginterp.html#autotoc_md171", null ],
            [ "Auto-tuning", "stage_ginterp.html#autotoc_md172", null ],
            [ "Error bound and limitations", "stage_ginterp.html#autotoc_md173", null ]
          ] ],
          [ "Ports", "stage_ginterp.html#autotoc_md175", [
            [ "Forward", "stage_ginterp.html#autotoc_md176", null ],
            [ "Inverse", "stage_ginterp.html#autotoc_md177", null ]
          ] ],
          [ "Typical pipeline", "stage_ginterp.html#autotoc_md179", null ],
          [ "TOML configuration", "stage_ginterp.html#autotoc_md181", null ],
          [ "Serialized header", "stage_ginterp.html#autotoc_md183", null ],
          [ "Acknowledgements", "stage_ginterp.html#autotoc_md185", null ]
        ] ],
        [ "BitplaneRZEStage", "stage_bitplane_rze.html", [
          [ "What it does", "stage_bitplane_rze.html#autotoc_md128", null ],
          [ "Relationship to BitshuffleStage + RZEStage", "stage_bitplane_rze.html#autotoc_md129", null ],
          [ "Template parameters", "stage_bitplane_rze.html#autotoc_md130", null ],
          [ "Stage settings", "stage_bitplane_rze.html#autotoc_md131", null ],
          [ "Graph compatibility", "stage_bitplane_rze.html#autotoc_md132", null ],
          [ "Ports", "stage_bitplane_rze.html#autotoc_md133", null ],
          [ "Typical pipeline", "stage_bitplane_rze.html#autotoc_md134", null ],
          [ "TOML configuration", "stage_bitplane_rze.html#autotoc_md135", null ],
          [ "Archive layout", "stage_bitplane_rze.html#autotoc_md136", null ],
          [ "Acknowledgements", "stage_bitplane_rze.html#autotoc_md137", null ]
        ] ]
      ] ],
      [ "Predictor stages", "stage_predictors.html", [
        [ "LorenzoStage", "stage_lorenzo.html", [
          [ "What it does", "stage_lorenzo.html#autotoc_md212", null ],
          [ "Template parameter", "stage_lorenzo.html#autotoc_md214", null ],
          [ "Available instantiations", "stage_lorenzo.html#autotoc_md215", null ],
          [ "Stage settings", "stage_lorenzo.html#autotoc_md217", null ],
          [ "Ports", "stage_lorenzo.html#autotoc_md219", null ],
          [ "Dimension setup — critical ordering rule", "stage_lorenzo.html#autotoc_md221", null ],
          [ "Typical pipeline (cuSZp-style)", "stage_lorenzo.html#autotoc_md223", null ]
        ] ],
        [ "TiledLorenzoStage", "stage_tiled_lorenzo.html", [
          [ "What it does", "stage_tiled_lorenzo.html#autotoc_md336", [
            [ "Tile-major output", "stage_tiled_lorenzo.html#autotoc_md337", null ]
          ] ],
          [ "Template parameter", "stage_tiled_lorenzo.html#autotoc_md339", null ],
          [ "Stage settings", "stage_tiled_lorenzo.html#autotoc_md341", null ],
          [ "Ports", "stage_tiled_lorenzo.html#autotoc_md343", null ],
          [ "Graph compatibility", "stage_tiled_lorenzo.html#autotoc_md345", null ],
          [ "Typical pipeline (cuSZp3)", "stage_tiled_lorenzo.html#autotoc_md347", null ],
          [ "TOML", "stage_tiled_lorenzo.html#autotoc_md349", null ],
          [ "Acknowledgements", "stage_tiled_lorenzo.html#autotoc_md351", null ]
        ] ],
        [ "DifferenceStage", "stage_diff.html", [
          [ "What it does", "stage_diff.html#autotoc_md149", null ],
          [ "Template parameters", "stage_diff.html#autotoc_md151", null ],
          [ "Available instantiations", "stage_diff.html#autotoc_md152", null ],
          [ "Stage settings", "stage_diff.html#autotoc_md154", null ],
          [ "Chunking", "stage_diff.html#autotoc_md156", null ],
          [ "Common instantiations", "stage_diff.html#autotoc_md157", null ],
          [ "Acknowledgements", "stage_diff.html#autotoc_md159", null ],
          [ "Typical pipeline", "stage_diff.html#autotoc_md161", null ]
        ] ]
      ] ],
      [ "Quantizer stages", "stage_quantizers.html", [
        [ "QuantizerStage", "stage_quantizer.html", [
          [ "What it does", "stage_quantizer.html#autotoc_md268", null ],
          [ "Template parameters", "stage_quantizer.html#autotoc_md270", null ],
          [ "Available instantiations", "stage_quantizer.html#autotoc_md271", null ],
          [ "Stage settings", "stage_quantizer.html#autotoc_md273", null ],
          [ "Output ports (compression)", "stage_quantizer.html#autotoc_md275", [
            [ "Normal mode (3 outputs)", "stage_quantizer.html#autotoc_md276", null ],
            [ "Inplace outlier mode (1 output)", "stage_quantizer.html#autotoc_md277", null ],
            [ "Linear / no-outlier mode (1 output)", "stage_quantizer.html#autotoc_md278", null ]
          ] ],
          [ "Linear / no-outlier mode (ABS/NOA only)", "stage_quantizer.html#autotoc_md280", null ],
          [ "Error bound modes", "stage_quantizer.html#autotoc_md282", null ],
          [ "Inplace outlier constraints (ABS/NOA only)", "stage_quantizer.html#autotoc_md285", [
            [ "1. Zigzag encoding must be enabled", "stage_quantizer.html#autotoc_md286", null ],
            [ "2. sizeof(TCode) == sizeof(TInput)", "stage_quantizer.html#autotoc_md287", null ],
            [ "Why REL does not support inplace outliers", "stage_quantizer.html#autotoc_md288", null ]
          ] ],
          [ "Value base and CUDA Graph capture", "stage_quantizer.html#autotoc_md290", null ],
          [ "Typical pipelines", "stage_quantizer.html#autotoc_md292", [
            [ "PFPL-style (standalone quantizer)", "stage_quantizer.html#autotoc_md293", null ],
            [ "Inplace outlier pipeline", "stage_quantizer.html#autotoc_md294", null ]
          ] ],
          [ "Acknowledgements", "stage_quantizer.html#autotoc_md296", null ]
        ] ]
      ] ],
      [ "Coder stages", "stage_coders.html", [
        [ "HuffmanStage", "stage_huffman.html", [
          [ "What it does", "stage_huffman.html#autotoc_md187", null ],
          [ "Template parameter", "stage_huffman.html#autotoc_md189", null ],
          [ "Available instantiations", "stage_huffman.html#autotoc_md190", null ],
          [ "Stage settings", "stage_huffman.html#autotoc_md192", [
            [ "Setting <tt>bklen</tt>", "stage_huffman.html#autotoc_md193", null ]
          ] ],
          [ "Typical pipeline", "stage_huffman.html#autotoc_md195", [
            [ "Standalone (symbol array input)", "stage_huffman.html#autotoc_md196", null ],
            [ "cuSZ-style Lorenzo + Huffman", "stage_huffman.html#autotoc_md197", null ]
          ] ],
          [ "TOML configuration", "stage_huffman.html#autotoc_md199", null ],
          [ "Execution flow (CPU–GPU movement pattern)", "stage_huffman.html#huffman-execution", [
            [ "Forward pass", "stage_huffman.html#autotoc_md201", null ],
            [ "Inverse pass", "stage_huffman.html#autotoc_md202", null ]
          ] ],
          [ "Internal buffer layout", "stage_huffman.html#autotoc_md204", null ],
          [ "Serialized header", "stage_huffman.html#autotoc_md206", null ],
          [ "Limitations", "stage_huffman.html#huffman-limitations", null ],
          [ "Acknowledgements", "stage_huffman.html#autotoc_md209", null ]
        ] ],
        [ "ANSStage", "stage_ans.html", [
          [ "What it does", "stage_ans.html#autotoc_md95", null ],
          [ "Stage settings", "stage_ans.html#autotoc_md97", [
            [ "prob_bits limitation", "stage_ans.html#autotoc_md98", null ]
          ] ],
          [ "Typical pipeline", "stage_ans.html#autotoc_md100", [
            [ "Standalone (byte array input)", "stage_ans.html#autotoc_md101", null ],
            [ "cuSZ-style Lorenzo + ANS", "stage_ans.html#autotoc_md102", null ]
          ] ],
          [ "TOML configuration", "stage_ans.html#autotoc_md104", null ],
          [ "Execution flow (CPU–GPU movement pattern)", "stage_ans.html#ans-execution", [
            [ "Forward pass", "stage_ans.html#autotoc_md106", null ],
            [ "Inverse pass", "stage_ans.html#autotoc_md107", null ]
          ] ],
          [ "Scratch buffers / device footprint", "stage_ans.html#autotoc_md109", null ],
          [ "Serialized header", "stage_ans.html#autotoc_md111", null ],
          [ "Limitations", "stage_ans.html#ans-limitations", null ],
          [ "Acknowledgements", "stage_ans.html#autotoc_md114", null ]
        ] ],
        [ "RLEStage", "stage_rle.html", [
          [ "What it does", "stage_rle.html#autotoc_md298", null ],
          [ "Template parameter", "stage_rle.html#autotoc_md300", null ],
          [ "Available instantiations", "stage_rle.html#autotoc_md301", null ],
          [ "Stage settings", "stage_rle.html#autotoc_md303", null ],
          [ "Typical pipeline", "stage_rle.html#autotoc_md306", null ],
          [ "Stream layout (forward output)", "stage_rle.html#autotoc_md308", null ]
        ] ],
        [ "RZEStage", "stage_rze.html", [
          [ "What it does", "stage_rze.html#autotoc_md324", null ],
          [ "Stage settings", "stage_rze.html#autotoc_md326", null ],
          [ "Alignment requirement", "stage_rze.html#autotoc_md328", null ],
          [ "Typical pipeline", "stage_rze.html#autotoc_md330", null ],
          [ "Stream layout (forward output)", "stage_rze.html#autotoc_md332", null ],
          [ "Acknowledgements", "stage_rze.html#autotoc_md334", null ]
        ] ],
        [ "RREStage", "stage_rre.html", [
          [ "What it does", "stage_rre.html#autotoc_md310", null ],
          [ "Stage settings", "stage_rre.html#autotoc_md312", null ],
          [ "Alignment requirement", "stage_rre.html#autotoc_md314", null ],
          [ "Graph capture", "stage_rre.html#autotoc_md316", null ],
          [ "Typical pipeline", "stage_rre.html#autotoc_md318", null ],
          [ "Stream layout (forward output)", "stage_rre.html#autotoc_md320", null ],
          [ "Acknowledgements", "stage_rre.html#autotoc_md322", null ]
        ] ],
        [ "BitpackStage", "stage_bitpack.html", [
          [ "What it does", "stage_bitpack.html#autotoc_md116", null ],
          [ "Template parameter", "stage_bitpack.html#autotoc_md118", null ],
          [ "Available instantiations", "stage_bitpack.html#autotoc_md119", null ],
          [ "Stage settings", "stage_bitpack.html#autotoc_md121", [
            [ "Manual bit-width", "stage_bitpack.html#autotoc_md122", null ],
            [ "Auto-detect mode", "stage_bitpack.html#autotoc_md123", null ]
          ] ],
          [ "Typical pipeline", "stage_bitpack.html#autotoc_md125", [
            [ "Manual <tt>nbits</tt>", "stage_bitpack.html#autotoc_md126", null ],
            [ "Auto-detect <tt>nbits</tt>", "stage_bitpack.html#autotoc_md127", null ]
          ] ]
        ] ],
        [ "AdaptiveBitpackStage", "stage_adaptive_bitpack.html", [
          [ "What it does", "stage_adaptive_bitpack.html#autotoc_md57", null ],
          [ "Template parameter", "stage_adaptive_bitpack.html#autotoc_md59", null ],
          [ "Stage settings", "stage_adaptive_bitpack.html#autotoc_md61", [
            [ "Outlier selection (cuSZp2)", "stage_adaptive_bitpack.html#autotoc_md62", [
              [ "Metadata difference from cuSZp2 (intentional)", "stage_adaptive_bitpack.html#autotoc_md63", null ]
            ] ]
          ] ],
          [ "Ports", "stage_adaptive_bitpack.html#autotoc_md65", null ],
          [ "Graph compatibility", "stage_adaptive_bitpack.html#autotoc_md67", null ],
          [ "Typical pipeline (cuSZp-style)", "stage_adaptive_bitpack.html#autotoc_md69", null ],
          [ "TOML", "stage_adaptive_bitpack.html#autotoc_md71", null ],
          [ "Acknowledgements", "stage_adaptive_bitpack.html#autotoc_md73", null ]
        ] ]
      ] ],
      [ "Shuffler stages", "stage_shufflers.html", [
        [ "BitshuffleStage", "stage_bitshuffle.html", [
          [ "What it does", "stage_bitshuffle.html#autotoc_md139", null ],
          [ "Stage settings", "stage_bitshuffle.html#autotoc_md141", null ],
          [ "Alignment requirement", "stage_bitshuffle.html#autotoc_md143", null ],
          [ "Typical pipeline", "stage_bitshuffle.html#autotoc_md145", null ],
          [ "Acknowledgements", "stage_bitshuffle.html#autotoc_md147", null ]
        ] ]
      ] ],
      [ "Transform stages", "stage_transforms.html", [
        [ "ZigzagStage", "stage_zigzag.html", [
          [ "What it does", "stage_zigzag.html#autotoc_md353", null ],
          [ "Template parameters", "stage_zigzag.html#autotoc_md355", null ],
          [ "Available instantiations", "stage_zigzag.html#autotoc_md356", null ],
          [ "Stage settings", "stage_zigzag.html#autotoc_md358", null ],
          [ "Ports", "stage_zigzag.html#autotoc_md360", null ],
          [ "Typical pipeline", "stage_zigzag.html#autotoc_md362", null ]
        ] ],
        [ "NegabinaryStage", "stage_negabinary.html", [
          [ "What it does", "stage_negabinary.html#autotoc_md257", null ],
          [ "Template parameters", "stage_negabinary.html#autotoc_md259", null ],
          [ "Available instantiations", "stage_negabinary.html#autotoc_md260", null ],
          [ "Stage settings", "stage_negabinary.html#autotoc_md262", null ],
          [ "Ports", "stage_negabinary.html#autotoc_md264", null ],
          [ "Typical pipeline", "stage_negabinary.html#autotoc_md266", null ]
        ] ],
        [ "ADMStage", "stage_adm.html", [
          [ "What it does", "stage_adm.html#autotoc_md75", null ],
          [ "Stage settings", "stage_adm.html#autotoc_md77", null ],
          [ "Typical pipeline", "stage_adm.html#autotoc_md79", [
            [ "Standalone (integer array input)", "stage_adm.html#autotoc_md80", null ],
            [ "cuSZ-style Lorenzo + ADM + ANS (recommended)", "stage_adm.html#autotoc_md81", null ]
          ] ],
          [ "TOML configuration", "stage_adm.html#autotoc_md83", null ],
          [ "Execution flow (CPU–GPU movement pattern)", "stage_adm.html#adm-execution", [
            [ "Forward pass", "stage_adm.html#autotoc_md85", null ],
            [ "Inverse pass", "stage_adm.html#autotoc_md86", null ]
          ] ],
          [ "Scratch buffers / device footprint", "stage_adm.html#autotoc_md88", null ],
          [ "Serialized header", "stage_adm.html#autotoc_md90", null ],
          [ "Limitations", "stage_adm.html#adm-limitations", null ],
          [ "Acknowledgements", "stage_adm.html#autotoc_md93", null ]
        ] ]
      ] ],
      [ "Structural stages", "stage_structural.html", [
        [ "MergeStage", "stage_merge.html", [
          [ "What it does", "stage_merge.html#autotoc_md244", null ],
          [ "Why it exists", "stage_merge.html#autotoc_md245", null ],
          [ "Stage settings", "stage_merge.html#autotoc_md247", null ],
          [ "Behaviour notes", "stage_merge.html#autotoc_md249", null ],
          [ "Typical pipeline (cuSZ-Hi cr-mode merged blob)", "stage_merge.html#autotoc_md251", null ],
          [ "TOML configuration", "stage_merge.html#autotoc_md253", null ],
          [ "Serialized config header", "stage_merge.html#autotoc_md255", null ]
        ] ]
      ] ]
    ] ],
    [ "Pipeline Configuration Files", "config_file_overview.html", [
      [ "API", "config_file_overview.html#autotoc_md364", [
        [ "Methods", "config_file_overview.html#autotoc_md365", null ],
        [ "Usage patterns", "config_file_overview.html#autotoc_md366", null ]
      ] ],
      [ "TOML Schema", "config_file_overview.html#autotoc_md368", [
        [ "[pipeline] – pipeline-level settings", "config_file_overview.html#autotoc_md369", null ],
        [ "[[stage]] – one entry per stage", "config_file_overview.html#autotoc_md370", null ]
      ] ],
      [ "Stage Types", "config_file_overview.html#autotoc_md372", [
        [ "Lorenzo1D / Lorenzo2D / Lorenzo3D", "config_file_overview.html#autotoc_md373", null ],
        [ "Bitshuffle", "config_file_overview.html#autotoc_md374", null ],
        [ "RZE", "config_file_overview.html#autotoc_md375", null ],
        [ "RRE", "config_file_overview.html#autotoc_md376", null ],
        [ "Merge", "config_file_overview.html#autotoc_md377", null ],
        [ "RLE", "config_file_overview.html#autotoc_md378", null ],
        [ "Difference", "config_file_overview.html#autotoc_md379", null ],
        [ "Zigzag", "config_file_overview.html#autotoc_md380", null ],
        [ "Quantizer", "config_file_overview.html#autotoc_md381", null ],
        [ "Negabinary", "config_file_overview.html#autotoc_md382", null ],
        [ "Bitpack", "config_file_overview.html#autotoc_md383", null ],
        [ "Huffman", "config_file_overview.html#autotoc_md384", null ]
      ] ],
      [ "Complete Examples", "config_file_overview.html#autotoc_md386", [
        [ "Lorenzo-based pipeline (ABS error)", "config_file_overview.html#autotoc_md387", null ],
        [ "PFPL pipeline (Quantizer, REL error)", "config_file_overview.html#autotoc_md388", null ]
      ] ],
      [ "Limitations", "config_file_overview.html#autotoc_md390", null ]
    ] ],
    [ "Command Line Interface", "cli_overview.html", [
      [ "Dynamic linear pipelines", "cli_overview.html#autotoc_md391", null ],
      [ "Decompress, compare, and report", "cli_overview.html#autotoc_md392", null ],
      [ "Branched pipelines via TOML config", "cli_overview.html#autotoc_md393", null ],
      [ "Benchmarking", "cli_overview.html#autotoc_md394", null ],
      [ "Machine-readable JSON reports", "cli_overview.html#autotoc_md395", null ],
      [ "Key flags", "cli_overview.html#autotoc_md396", null ]
    ] ],
    [ "API Reference", "api_reference.html", [
      [ "Lifecycle at a Glance", "api_reference.html#autotoc_md398", null ],
      [ "Enums", "api_reference.html#autotoc_md400", [
        [ "fz::MemoryStrategy", "api_reference.html#autotoc_md401", null ],
        [ "fz::ErrorBoundMode", "api_reference.html#autotoc_md402", null ]
      ] ],
      [ "Construction", "api_reference.html#autotoc_md404", null ],
      [ "Configuration (before finalize())", "api_reference.html#autotoc_md406", null ],
      [ "Building the Graph", "api_reference.html#autotoc_md408", null ],
      [ "Compression", "api_reference.html#autotoc_md410", [
        [ "Pool-owned output (default)", "api_reference.html#autotoc_md411", null ],
        [ "Caller-owned output", "api_reference.html#autotoc_md412", null ]
      ] ],
      [ "Decompression", "api_reference.html#autotoc_md414", [
        [ "Pool-owned output (default)", "api_reference.html#autotoc_md415", null ],
        [ "Caller-owned output", "api_reference.html#autotoc_md416", null ],
        [ "Caller-allocated buffer (no internal allocation)", "api_reference.html#autotoc_md417", null ],
        [ "Stream-concurrent decode (<tt>decompressInto</tt>, async)", "api_reference.html#autotoc_md418", null ]
      ] ],
      [ "Memory Ownership Summary", "api_reference.html#autotoc_md420", null ],
      [ "File I/O", "api_reference.html#autotoc_md422", null ],
      [ "Decode-only pipelines (no warmup compress)", "api_reference.html#autotoc_md424", null ],
      [ "CUDA Graph Capture", "api_reference.html#autotoc_md426", null ],
      [ "Diagnostics", "api_reference.html#autotoc_md428", null ],
      [ "Common Gotchas", "api_reference.html#autotoc_md430", null ],
      [ "API Stability and Versioning", "api_reference.html#api_stability", [
        [ "Public API boundary", "api_reference.html#autotoc_md432", null ],
        [ "Versioning policy (SemVer)", "api_reference.html#autotoc_md433", null ],
        [ "Stage interface stability", "api_reference.html#autotoc_md434", null ],
        [ "API change checklist", "api_reference.html#autotoc_md435", null ]
      ] ]
    ] ],
    [ "Architecture Overview", "architecture.html", [
      [ "Design Goals", "architecture.html#autotoc_md437", null ],
      [ "Layer Model", "architecture.html#autotoc_md439", null ],
      [ "Key Abstractions", "architecture.html#autotoc_md441", [
        [ "Stage", "architecture.html#autotoc_md442", null ],
        [ "Pipeline", "architecture.html#autotoc_md443", null ],
        [ "CompressionDAG", "architecture.html#autotoc_md444", null ],
        [ "MemoryPool", "architecture.html#autotoc_md445", null ]
      ] ],
      [ "Execution Flow", "architecture.html#autotoc_md447", [
        [ "Compression", "architecture.html#autotoc_md448", null ],
        [ "Decompression", "architecture.html#autotoc_md449", null ]
      ] ],
      [ "Memory Ownership", "architecture.html#autotoc_md451", null ],
      [ "Logging", "architecture.html#autotoc_md453", null ],
      [ "Related Pages", "architecture.html#autotoc_md455", null ]
    ] ],
    [ "How to Add a New Stage", "how_to_add_a_stage.html", [
      [ "Overview", "how_to_add_a_stage.html#autotoc_md457", null ],
      [ "Step 1 — Choose a location", "how_to_add_a_stage.html#autotoc_md459", [
        [ "Category definitions", "how_to_add_a_stage.html#autotoc_md460", null ]
      ] ],
      [ "Step 2 — Write the header (<name>_stage.h)", "how_to_add_a_stage.html#autotoc_md462", [
        [ "Multi-output stages", "how_to_add_a_stage.html#autotoc_md463", null ],
        [ "Non-size-preserving stages: bidirectional estimateOutputSizes", "how_to_add_a_stage.html#autotoc_md464", null ],
        [ "Persistent scratch memory", "how_to_add_a_stage.html#autotoc_md465", null ],
        [ "CUDA Graph compatibility", "how_to_add_a_stage.html#autotoc_md466", null ],
        [ "Input alignment", "how_to_add_a_stage.html#autotoc_md467", null ]
      ] ],
      [ "Step 3 — Write the implementation (<name>_stage.cu)", "how_to_add_a_stage.html#autotoc_md469", [
        [ "Shared output locations", "how_to_add_a_stage.html#autotoc_md470", null ]
      ] ],
      [ "Step 4 — Register the StageType", "how_to_add_a_stage.html#autotoc_md472", null ],
      [ "Step 5 — Register in the factory", "how_to_add_a_stage.html#autotoc_md474", null ],
      [ "Step 6 — Add to CMakeLists.txt", "how_to_add_a_stage.html#autotoc_md476", null ],
      [ "Step 6b — Export in the public header", "how_to_add_a_stage.html#autotoc_md478", null ],
      [ "Step 7 — Register in the TOML config loader", "how_to_add_a_stage.html#autotoc_md480", null ],
      [ "Step 8 — Register in the CLI dynamic builder *(optional)*", "how_to_add_a_stage.html#autotoc_md482", null ],
      [ "Step 8b — Attribution *(required when based on prior work)*", "how_to_add_a_stage.html#autotoc_md484", null ],
      [ "Step 9 — Write tests", "how_to_add_a_stage.html#autotoc_md486", null ],
      [ "Checklist", "how_to_add_a_stage.html#autotoc_md488", null ]
    ] ],
    [ "FZM File Format", "fzm_format.html", [
      [ "Version History", "fzm_format.html#autotoc_md490", null ],
      [ "File Layout", "fzm_format.html#autotoc_md492", null ],
      [ "FZMHeaderCore (80 bytes)", "fzm_format.html#autotoc_md494", null ],
      [ "FZMStageInfo (256 bytes, one per stage)", "fzm_format.html#autotoc_md496", null ],
      [ "FZMBufferEntry (256 bytes, one per buffer)", "fzm_format.html#autotoc_md498", null ],
      [ "StageType Values", "fzm_format.html#autotoc_md500", null ],
      [ "DataType Values", "fzm_format.html#autotoc_md502", null ],
      [ "Reading a File Without the Library", "fzm_format.html#autotoc_md504", null ],
      [ "Versioning Rules", "fzm_format.html#autotoc_md506", null ]
    ] ],
    [ "Docker Setup", "md_docs_2docker.html", [
      [ "Overview", "md_docs_2docker.html#autotoc_md508", null ],
      [ "Building the Docker Image", "md_docs_2docker.html#autotoc_md509", null ],
      [ "Using the Pre-Installed Library", "md_docs_2docker.html#autotoc_md510", [
        [ "Quick Start", "md_docs_2docker.html#autotoc_md511", null ],
        [ "With CMake (Recommended)", "md_docs_2docker.html#autotoc_md512", null ],
        [ "Interactive Shell", "md_docs_2docker.html#autotoc_md513", null ]
      ] ],
      [ "Local Development (Building FZGPUModules Itself)", "md_docs_2docker.html#autotoc_md514", null ],
      [ "CI/CD Testing", "md_docs_2docker.html#autotoc_md515", [
        [ "Running the Test Suite", "md_docs_2docker.html#autotoc_md516", null ],
        [ "Full Build with All Targets", "md_docs_2docker.html#autotoc_md517", null ]
      ] ],
      [ "GPU Support", "md_docs_2docker.html#autotoc_md518", null ],
      [ "Development Notes", "md_docs_2docker.html#autotoc_md519", [
        [ "Sanitizers", "md_docs_2docker.html#autotoc_md520", null ],
        [ "Python Integration", "md_docs_2docker.html#autotoc_md521", null ]
      ] ],
      [ "Troubleshooting", "md_docs_2docker.html#autotoc_md522", [
        [ "GPU Not Detected", "md_docs_2docker.html#autotoc_md523", null ],
        [ "find_package Cannot Find FZGPUModules", "md_docs_2docker.html#autotoc_md524", null ],
        [ "Build Failures in CI", "md_docs_2docker.html#autotoc_md525", null ]
      ] ],
      [ "See Also", "md_docs_2docker.html#autotoc_md526", null ]
    ] ],
    [ "LibPressio Python Bindings", "libpressio_python.html", [
      [ "Setup", "libpressio_python.html#autotoc_md528", [
        [ "Prerequisites", "libpressio_python.html#autotoc_md529", null ],
        [ "Install spack", "libpressio_python.html#autotoc_md530", null ],
        [ "Add the spack package repos", "libpressio_python.html#autotoc_md531", null ],
        [ "Create and activate a spack environment", "libpressio_python.html#autotoc_md532", null ],
        [ "Point spack at the libpressio source fork", "libpressio_python.html#autotoc_md533", null ],
        [ "Install", "libpressio_python.html#autotoc_md534", null ],
        [ "Activate in Python", "libpressio_python.html#autotoc_md535", null ]
      ] ],
      [ "Quick Start", "libpressio_python.html#autotoc_md537", null ],
      [ "from_config Structure", "libpressio_python.html#autotoc_md539", null ],
      [ "Encode and Decode", "libpressio_python.html#autotoc_md541", null ],
      [ "Pipeline Options", "libpressio_python.html#autotoc_md543", [
        [ "Error bound modes", "libpressio_python.html#autotoc_md544", null ],
        [ "Connections format", "libpressio_python.html#autotoc_md545", null ]
      ] ],
      [ "Stage Tokens", "libpressio_python.html#autotoc_md547", [
        [ "Lorenzo Predictor + Quantizer", "libpressio_python.html#autotoc_md548", null ],
        [ "Standalone Quantizer", "libpressio_python.html#autotoc_md549", null ],
        [ "Difference Stage", "libpressio_python.html#autotoc_md550", null ],
        [ "Zigzag and Negabinary Transforms", "libpressio_python.html#autotoc_md551", null ],
        [ "Run-Length Encoding (RLE)", "libpressio_python.html#autotoc_md552", null ],
        [ "Bitpacking", "libpressio_python.html#autotoc_md553", null ],
        [ "Bitshuffle", "libpressio_python.html#autotoc_md554", null ],
        [ "Repeated Zero Elimination (RZE)", "libpressio_python.html#autotoc_md555", null ],
        [ "Huffman Entropy Coding", "libpressio_python.html#autotoc_md556", null ]
      ] ],
      [ "Metrics", "libpressio_python.html#autotoc_md558", null ],
      [ "Common Recipes", "libpressio_python.html#autotoc_md560", [
        [ "Lorenzo + RLE (default)", "libpressio_python.html#autotoc_md561", null ],
        [ "Lorenzo + RZE (best ratio on smooth data)", "libpressio_python.html#autotoc_md562", null ],
        [ "Lorenzo + Bitshuffle", "libpressio_python.html#autotoc_md563", null ],
        [ "Quantizer with Inplace Outliers (float32 only)", "libpressio_python.html#autotoc_md564", null ],
        [ "Lossless Integer Lorenzo", "libpressio_python.html#autotoc_md565", null ],
        [ "3-D Structured Grid", "libpressio_python.html#autotoc_md566", null ]
      ] ],
      [ "CUDA Graph Mode", "libpressio_python.html#autotoc_md568", null ],
      [ "Exposing Stage Outputs", "libpressio_python.html#autotoc_md570", [
        [ "Stage output port names", "libpressio_python.html#autotoc_md571", null ]
      ] ],
      [ "TOML Config File", "libpressio_python.html#autotoc_md573", null ],
      [ "Error Handling", "libpressio_python.html#autotoc_md575", null ]
    ] ],
    [ "Acknowledgements", "acknowledgements.html", [
      [ "Summary", "acknowledgements.html#autotoc_md577", null ],
      [ "LC Framework", "acknowledgements.html#autotoc_md579", null ],
      [ "cuSZ / PHF", "acknowledgements.html#autotoc_md581", null ],
      [ "FZ-GPU", "acknowledgements.html#autotoc_md583", null ],
      [ "cuSZ-Hi", "acknowledgements.html#autotoc_md585", null ],
      [ "cuSZp / cuSZp2 / cuSZp3", "acknowledgements.html#autotoc_md587", null ],
      [ "MANS", "acknowledgements.html#autotoc_md589", null ],
      [ "dietGPU", "acknowledgements.html#autotoc_md591", null ]
    ] ],
    [ "Namespace Members", "namespacemembers.html", [
      [ "All", "namespacemembers.html", null ],
      [ "Functions", "namespacemembers_func.html", null ],
      [ "Variables", "namespacemembers_vars.html", null ],
      [ "Enumerations", "namespacemembers_enum.html", null ]
    ] ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", null ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"BatchPrefixSum_8h_source.html",
"classfz_1_1LorenzoQuantStage.html#a601caab47be8ee0a1729ad6c2c93bd12",
"config_8h.html",
"libpressio_python.html#autotoc_md568",
"stage_rre.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';