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
    [ "Overview", "index.html#autotoc_md0", [
      [ "Requirements", "index.html#autotoc_md2", null ]
    ] ],
    [ "Quick Start", "index.html#autotoc_md4", [
      [ "Building from Source", "index.html#autotoc_md5", null ],
      [ "C++ API Usage", "index.html#autotoc_md6", null ],
      [ "Available Stages", "index.html#mainpage_stages", null ],
      [ "Memory Strategies", "index.html#autotoc_md8", null ],
      [ "Caller-Allocated Output", "index.html#autotoc_md10", null ],
      [ "CUDA Graph Support", "index.html#autotoc_md12", null ],
      [ "Compressor Config File", "index.html#autotoc_md14", null ],
      [ "File I/O", "index.html#autotoc_md16", null ],
      [ "Thread Safety", "index.html#autotoc_md18", null ]
    ] ],
    [ "Citation", "index.html#autotoc_md20", null ],
    [ "Building from Source", "building_from_source.html", [
      [ "Prerequisites", "building_from_source.html#autotoc_md22", null ],
      [ "Getting the Source", "building_from_source.html#autotoc_md24", null ],
      [ "Quick Build", "building_from_source.html#autotoc_md26", null ],
      [ "CMake Presets", "building_from_source.html#autotoc_md28", null ],
      [ "CMake Options", "building_from_source.html#autotoc_md30", null ],
      [ "Common Build Examples", "building_from_source.html#autotoc_md32", null ],
      [ "Binary Output", "building_from_source.html#autotoc_md34", null ],
      [ "Testing", "building_from_source.html#autotoc_md36", [
        [ "Host Sanitizers (ASan + UBSan)", "building_from_source.html#autotoc_md37", null ],
        [ "CUDA Compute Sanitizer", "building_from_source.html#autotoc_md38", null ],
        [ "ThreadSanitizer", "building_from_source.html#autotoc_md39", null ]
      ] ],
      [ "Install", "building_from_source.html#autotoc_md41", null ],
      [ "Using from CMake", "building_from_source.html#autotoc_md43", null ],
      [ "Generating Documentation", "building_from_source.html#autotoc_md45", null ]
    ] ],
    [ "Stage Reference", "stages_overview.html", [
      [ "Categories", "stages_overview.html#autotoc_md101", null ],
      [ "Fused stages", "stage_fused.html", [
        [ "LorenzoQuantStage", "stage_lorenzo_quant.html", [
          [ "What it does", "stage_lorenzo_quant.html#autotoc_md116", null ],
          [ "Template parameters", "stage_lorenzo_quant.html#autotoc_md118", null ],
          [ "Available instantiations", "stage_lorenzo_quant.html#autotoc_md119", null ],
          [ "Stage settings", "stage_lorenzo_quant.html#autotoc_md121", null ],
          [ "Output ports (compression)", "stage_lorenzo_quant.html#autotoc_md123", null ],
          [ "Error bound modes", "stage_lorenzo_quant.html#autotoc_md125", null ],
          [ "Dimension setup — critical ordering rule", "stage_lorenzo_quant.html#autotoc_md127", null ],
          [ "Value base and CUDA Graph capture", "stage_lorenzo_quant.html#autotoc_md129", null ],
          [ "Typical pipeline", "stage_lorenzo_quant.html#autotoc_md131", null ]
        ] ]
      ] ],
      [ "Predictor stages", "stage_predictors.html", [
        [ "LorenzoStage", "stage_lorenzo.html", [
          [ "What it does", "stage_lorenzo.html#autotoc_md103", null ],
          [ "Template parameter", "stage_lorenzo.html#autotoc_md105", null ],
          [ "Available instantiations", "stage_lorenzo.html#autotoc_md106", null ],
          [ "Stage settings", "stage_lorenzo.html#autotoc_md108", null ],
          [ "Ports", "stage_lorenzo.html#autotoc_md110", null ],
          [ "Dimension setup — critical ordering rule", "stage_lorenzo.html#autotoc_md112", null ],
          [ "Typical pipeline (cuSZp-style)", "stage_lorenzo.html#autotoc_md114", null ]
        ] ],
        [ "DifferenceStage", "stage_diff.html", [
          [ "What it does", "stage_diff.html#autotoc_md68", null ],
          [ "Template parameters", "stage_diff.html#autotoc_md70", null ],
          [ "Available instantiations", "stage_diff.html#autotoc_md71", null ],
          [ "Stage settings", "stage_diff.html#autotoc_md73", null ],
          [ "Chunking", "stage_diff.html#autotoc_md75", null ],
          [ "Common instantiations", "stage_diff.html#autotoc_md76", null ],
          [ "Typical pipeline", "stage_diff.html#autotoc_md78", null ]
        ] ]
      ] ],
      [ "Quantizer stages", "stage_quantizers.html", [
        [ "QuantizerStage", "stage_quantizer.html", [
          [ "What it does", "stage_quantizer.html#autotoc_md144", null ],
          [ "Template parameters", "stage_quantizer.html#autotoc_md146", null ],
          [ "Available instantiations", "stage_quantizer.html#autotoc_md147", null ],
          [ "Stage settings", "stage_quantizer.html#autotoc_md149", null ],
          [ "Output ports (compression)", "stage_quantizer.html#autotoc_md151", [
            [ "Normal mode (4 outputs)", "stage_quantizer.html#autotoc_md152", null ],
            [ "Inplace outlier mode (1 output)", "stage_quantizer.html#autotoc_md153", null ]
          ] ],
          [ "Error bound modes", "stage_quantizer.html#autotoc_md155", null ],
          [ "Inplace outlier constraints (ABS/NOA only)", "stage_quantizer.html#autotoc_md158", [
            [ "1. Zigzag encoding must be enabled", "stage_quantizer.html#autotoc_md159", null ],
            [ "2. sizeof(TCode) == sizeof(TInput)", "stage_quantizer.html#autotoc_md160", null ],
            [ "Why REL does not support inplace outliers", "stage_quantizer.html#autotoc_md161", null ]
          ] ],
          [ "Value base and CUDA Graph capture", "stage_quantizer.html#autotoc_md163", null ],
          [ "Typical pipelines", "stage_quantizer.html#autotoc_md165", [
            [ "PFPL-style (standalone quantizer)", "stage_quantizer.html#autotoc_md166", null ],
            [ "Inplace outlier pipeline", "stage_quantizer.html#autotoc_md167", null ]
          ] ]
        ] ]
      ] ],
      [ "Coder stages", "stage_coders.html", [
        [ "HuffmanStage", "stage_huffman.html", [
          [ "What it does", "stage_huffman.html#autotoc_md80", null ],
          [ "Template parameter", "stage_huffman.html#autotoc_md82", null ],
          [ "Available instantiations", "stage_huffman.html#autotoc_md83", null ],
          [ "Stage settings", "stage_huffman.html#autotoc_md85", [
            [ "Setting <tt>bklen</tt>", "stage_huffman.html#autotoc_md86", null ]
          ] ],
          [ "Typical pipeline", "stage_huffman.html#autotoc_md88", [
            [ "Standalone (symbol array input)", "stage_huffman.html#autotoc_md89", null ],
            [ "cuSZ-style Lorenzo + Huffman", "stage_huffman.html#autotoc_md90", null ]
          ] ],
          [ "TOML configuration", "stage_huffman.html#autotoc_md92", null ],
          [ "Execution flow (CPU–GPU movement pattern)", "stage_huffman.html#huffman-execution", [
            [ "Forward pass", "stage_huffman.html#autotoc_md94", null ],
            [ "Inverse pass", "stage_huffman.html#autotoc_md95", null ]
          ] ],
          [ "Internal buffer layout", "stage_huffman.html#autotoc_md97", null ],
          [ "Serialized header", "stage_huffman.html#autotoc_md99", null ],
          [ "Limitations", "stage_huffman.html#huffman-limitations", null ]
        ] ],
        [ "RLEStage", "stage_rle.html", [
          [ "What it does", "stage_rle.html#autotoc_md169", null ],
          [ "Template parameter", "stage_rle.html#autotoc_md171", null ],
          [ "Available instantiations", "stage_rle.html#autotoc_md172", null ],
          [ "Stage settings", "stage_rle.html#autotoc_md174", null ],
          [ "Typical pipeline", "stage_rle.html#autotoc_md177", null ],
          [ "Stream layout (forward output)", "stage_rle.html#autotoc_md179", null ]
        ] ],
        [ "RZEStage", "stage_rze.html", [
          [ "What it does", "stage_rze.html#autotoc_md181", null ],
          [ "Stage settings", "stage_rze.html#autotoc_md183", null ],
          [ "Alignment requirement", "stage_rze.html#autotoc_md185", null ],
          [ "Typical pipeline", "stage_rze.html#autotoc_md187", null ],
          [ "Stream layout (forward output)", "stage_rze.html#autotoc_md189", null ]
        ] ],
        [ "BitpackStage", "stage_bitpack.html", [
          [ "What it does", "stage_bitpack.html#autotoc_md47", null ],
          [ "Template parameter", "stage_bitpack.html#autotoc_md49", null ],
          [ "Available instantiations", "stage_bitpack.html#autotoc_md50", null ],
          [ "Stage settings", "stage_bitpack.html#autotoc_md52", [
            [ "Manual bit-width", "stage_bitpack.html#autotoc_md53", null ],
            [ "Auto-detect mode", "stage_bitpack.html#autotoc_md54", null ]
          ] ],
          [ "Typical pipeline", "stage_bitpack.html#autotoc_md56", [
            [ "Manual <tt>nbits</tt>", "stage_bitpack.html#autotoc_md57", null ],
            [ "Auto-detect <tt>nbits</tt>", "stage_bitpack.html#autotoc_md58", null ]
          ] ]
        ] ]
      ] ],
      [ "Shuffler stages", "stage_shufflers.html", [
        [ "BitshuffleStage", "stage_bitshuffle.html", [
          [ "What it does", "stage_bitshuffle.html#autotoc_md60", null ],
          [ "Stage settings", "stage_bitshuffle.html#autotoc_md62", null ],
          [ "Alignment requirement", "stage_bitshuffle.html#autotoc_md64", null ],
          [ "Typical pipeline", "stage_bitshuffle.html#autotoc_md66", null ]
        ] ]
      ] ],
      [ "Transform stages", "stage_transforms.html", [
        [ "ZigzagStage", "stage_zigzag.html", [
          [ "What it does", "stage_zigzag.html#autotoc_md191", null ],
          [ "Template parameters", "stage_zigzag.html#autotoc_md193", null ],
          [ "Available instantiations", "stage_zigzag.html#autotoc_md194", null ],
          [ "Stage settings", "stage_zigzag.html#autotoc_md196", null ],
          [ "Ports", "stage_zigzag.html#autotoc_md198", null ],
          [ "Typical pipeline", "stage_zigzag.html#autotoc_md200", null ]
        ] ],
        [ "NegabinaryStage", "stage_negabinary.html", [
          [ "What it does", "stage_negabinary.html#autotoc_md133", null ],
          [ "Template parameters", "stage_negabinary.html#autotoc_md135", null ],
          [ "Available instantiations", "stage_negabinary.html#autotoc_md136", null ],
          [ "Stage settings", "stage_negabinary.html#autotoc_md138", null ],
          [ "Ports", "stage_negabinary.html#autotoc_md140", null ],
          [ "Typical pipeline", "stage_negabinary.html#autotoc_md142", null ]
        ] ]
      ] ]
    ] ],
    [ "Pipeline Configuration Files", "config_file_overview.html", [
      [ "API", "config_file_overview.html#autotoc_md202", [
        [ "Methods", "config_file_overview.html#autotoc_md203", null ],
        [ "Usage patterns", "config_file_overview.html#autotoc_md204", null ]
      ] ],
      [ "TOML Schema", "config_file_overview.html#autotoc_md206", [
        [ "[pipeline] – pipeline-level settings", "config_file_overview.html#autotoc_md207", null ],
        [ "[[stage]] – one entry per stage", "config_file_overview.html#autotoc_md208", null ]
      ] ],
      [ "Stage Types", "config_file_overview.html#autotoc_md210", [
        [ "Lorenzo1D / Lorenzo2D / Lorenzo3D", "config_file_overview.html#autotoc_md211", null ],
        [ "Bitshuffle", "config_file_overview.html#autotoc_md212", null ],
        [ "RZE", "config_file_overview.html#autotoc_md213", null ],
        [ "RLE", "config_file_overview.html#autotoc_md214", null ],
        [ "Difference", "config_file_overview.html#autotoc_md215", null ],
        [ "Zigzag", "config_file_overview.html#autotoc_md216", null ],
        [ "Quantizer", "config_file_overview.html#autotoc_md217", null ],
        [ "Negabinary", "config_file_overview.html#autotoc_md218", null ],
        [ "Bitpack", "config_file_overview.html#autotoc_md219", null ],
        [ "Huffman", "config_file_overview.html#autotoc_md220", null ]
      ] ],
      [ "Complete Examples", "config_file_overview.html#autotoc_md222", [
        [ "Lorenzo-based pipeline (ABS error)", "config_file_overview.html#autotoc_md223", null ],
        [ "PFPL pipeline (Quantizer, REL error)", "config_file_overview.html#autotoc_md224", null ]
      ] ],
      [ "Limitations", "config_file_overview.html#autotoc_md226", null ]
    ] ],
    [ "Command Line Interface", "cli_overview.html", [
      [ "Dynamic linear pipelines", "cli_overview.html#autotoc_md227", null ],
      [ "Decompress, compare, and report", "cli_overview.html#autotoc_md228", null ],
      [ "Branched pipelines via TOML config", "cli_overview.html#autotoc_md229", null ],
      [ "Benchmarking", "cli_overview.html#autotoc_md230", null ],
      [ "Key flags", "cli_overview.html#autotoc_md231", null ]
    ] ],
    [ "API Reference", "api_reference.html", [
      [ "Lifecycle at a Glance", "api_reference.html#autotoc_md233", null ],
      [ "Enums", "api_reference.html#autotoc_md235", [
        [ "fz::MemoryStrategy", "api_reference.html#autotoc_md236", null ],
        [ "fz::ErrorBoundMode", "api_reference.html#autotoc_md237", null ]
      ] ],
      [ "Construction", "api_reference.html#autotoc_md239", null ],
      [ "Configuration (before finalize())", "api_reference.html#autotoc_md241", null ],
      [ "Building the Graph", "api_reference.html#autotoc_md243", null ],
      [ "Compression", "api_reference.html#autotoc_md245", [
        [ "Pool-owned output (default)", "api_reference.html#autotoc_md246", null ],
        [ "Caller-owned output", "api_reference.html#autotoc_md247", null ]
      ] ],
      [ "Decompression", "api_reference.html#autotoc_md249", [
        [ "Pool-owned output (default)", "api_reference.html#autotoc_md250", null ],
        [ "Caller-owned output", "api_reference.html#autotoc_md251", null ],
        [ "Caller-allocated buffer (no internal allocation)", "api_reference.html#autotoc_md252", null ]
      ] ],
      [ "Memory Ownership Summary", "api_reference.html#autotoc_md254", null ],
      [ "File I/O", "api_reference.html#autotoc_md256", null ],
      [ "CUDA Graph Capture", "api_reference.html#autotoc_md258", null ],
      [ "Diagnostics", "api_reference.html#autotoc_md260", null ],
      [ "Common Gotchas", "api_reference.html#autotoc_md262", null ],
      [ "API Stability and Versioning", "api_reference.html#api_stability", [
        [ "Public API boundary", "api_reference.html#autotoc_md264", null ],
        [ "Versioning policy (SemVer)", "api_reference.html#autotoc_md265", null ],
        [ "Stage interface stability", "api_reference.html#autotoc_md266", null ],
        [ "API change checklist", "api_reference.html#autotoc_md267", null ]
      ] ]
    ] ],
    [ "Architecture Overview", "architecture.html", [
      [ "Design Goals", "architecture.html#autotoc_md269", null ],
      [ "Layer Model", "architecture.html#autotoc_md271", null ],
      [ "Key Abstractions", "architecture.html#autotoc_md273", [
        [ "Stage", "architecture.html#autotoc_md274", null ],
        [ "Pipeline", "architecture.html#autotoc_md275", null ],
        [ "CompressionDAG", "architecture.html#autotoc_md276", null ],
        [ "MemoryPool", "architecture.html#autotoc_md277", null ]
      ] ],
      [ "Execution Flow", "architecture.html#autotoc_md279", [
        [ "Compression", "architecture.html#autotoc_md280", null ],
        [ "Decompression", "architecture.html#autotoc_md281", null ]
      ] ],
      [ "Memory Ownership", "architecture.html#autotoc_md283", null ],
      [ "Logging", "architecture.html#autotoc_md285", null ],
      [ "Related Pages", "architecture.html#autotoc_md287", null ]
    ] ],
    [ "How to Add a New Stage", "how_to_add_a_stage.html", [
      [ "Overview", "how_to_add_a_stage.html#autotoc_md289", null ],
      [ "Step 1 — Choose a location", "how_to_add_a_stage.html#autotoc_md291", [
        [ "Category definitions", "how_to_add_a_stage.html#autotoc_md292", null ]
      ] ],
      [ "Step 2 — Write the header (<name>_stage.h)", "how_to_add_a_stage.html#autotoc_md294", [
        [ "Multi-output stages", "how_to_add_a_stage.html#autotoc_md295", null ],
        [ "Non-size-preserving stages: bidirectional estimateOutputSizes", "how_to_add_a_stage.html#autotoc_md296", null ],
        [ "Persistent scratch memory", "how_to_add_a_stage.html#autotoc_md297", null ],
        [ "CUDA Graph compatibility", "how_to_add_a_stage.html#autotoc_md298", null ],
        [ "Input alignment", "how_to_add_a_stage.html#autotoc_md299", null ]
      ] ],
      [ "Step 3 — Write the implementation (<name>_stage.cu)", "how_to_add_a_stage.html#autotoc_md301", [
        [ "Shared output locations", "how_to_add_a_stage.html#autotoc_md302", null ]
      ] ],
      [ "Step 4 — Register the StageType", "how_to_add_a_stage.html#autotoc_md304", null ],
      [ "Step 5 — Register in the factory", "how_to_add_a_stage.html#autotoc_md306", null ],
      [ "Step 6 — Add to CMakeLists.txt", "how_to_add_a_stage.html#autotoc_md308", null ],
      [ "Step 6b — Export in the public header", "how_to_add_a_stage.html#autotoc_md310", null ],
      [ "Step 7 — Register in the TOML config loader", "how_to_add_a_stage.html#autotoc_md312", null ],
      [ "Step 8 — Register in the CLI dynamic builder *(optional)*", "how_to_add_a_stage.html#autotoc_md314", null ],
      [ "Step 9 — Write tests", "how_to_add_a_stage.html#autotoc_md316", null ],
      [ "Checklist", "how_to_add_a_stage.html#autotoc_md318", null ]
    ] ],
    [ "FZM File Format", "fzm_format.html", [
      [ "Version History", "fzm_format.html#autotoc_md320", null ],
      [ "File Layout", "fzm_format.html#autotoc_md322", null ],
      [ "FZMHeaderCore (80 bytes)", "fzm_format.html#autotoc_md324", null ],
      [ "FZMStageInfo (256 bytes, one per stage)", "fzm_format.html#autotoc_md326", null ],
      [ "FZMBufferEntry (256 bytes, one per buffer)", "fzm_format.html#autotoc_md328", null ],
      [ "StageType Values", "fzm_format.html#autotoc_md330", null ],
      [ "DataType Values", "fzm_format.html#autotoc_md332", null ],
      [ "Reading a File Without the Library", "fzm_format.html#autotoc_md334", null ],
      [ "Versioning Rules", "fzm_format.html#autotoc_md336", null ]
    ] ],
    [ "Docker Setup", "md_docs_2docker.html", [
      [ "Overview", "md_docs_2docker.html#autotoc_md338", null ],
      [ "Building the Docker Image", "md_docs_2docker.html#autotoc_md339", null ],
      [ "Using the Pre-Installed Library", "md_docs_2docker.html#autotoc_md340", [
        [ "Quick Start", "md_docs_2docker.html#autotoc_md341", null ],
        [ "With CMake (Recommended)", "md_docs_2docker.html#autotoc_md342", null ],
        [ "Interactive Shell", "md_docs_2docker.html#autotoc_md343", null ]
      ] ],
      [ "Local Development (Building FZGPUModules Itself)", "md_docs_2docker.html#autotoc_md344", null ],
      [ "CI/CD Testing", "md_docs_2docker.html#autotoc_md345", [
        [ "Running the Test Suite", "md_docs_2docker.html#autotoc_md346", null ],
        [ "Full Build with All Targets", "md_docs_2docker.html#autotoc_md347", null ]
      ] ],
      [ "GPU Support", "md_docs_2docker.html#autotoc_md348", null ],
      [ "Development Notes", "md_docs_2docker.html#autotoc_md349", [
        [ "Sanitizers", "md_docs_2docker.html#autotoc_md350", null ],
        [ "Python Integration", "md_docs_2docker.html#autotoc_md351", null ]
      ] ],
      [ "Troubleshooting", "md_docs_2docker.html#autotoc_md352", [
        [ "GPU Not Detected", "md_docs_2docker.html#autotoc_md353", null ],
        [ "find_package Cannot Find FZGPUModules", "md_docs_2docker.html#autotoc_md354", null ],
        [ "Build Failures in CI", "md_docs_2docker.html#autotoc_md355", null ]
      ] ],
      [ "See Also", "md_docs_2docker.html#autotoc_md356", null ]
    ] ],
    [ "LibPressio Python Bindings", "libpressio_python.html", [
      [ "Setup", "libpressio_python.html#autotoc_md358", [
        [ "Prerequisites", "libpressio_python.html#autotoc_md359", null ],
        [ "Install spack", "libpressio_python.html#autotoc_md360", null ],
        [ "Add the spack package repos", "libpressio_python.html#autotoc_md361", null ],
        [ "Create and activate a spack environment", "libpressio_python.html#autotoc_md362", null ],
        [ "Point spack at the libpressio source fork", "libpressio_python.html#autotoc_md363", null ],
        [ "Install", "libpressio_python.html#autotoc_md364", null ],
        [ "Activate in Python", "libpressio_python.html#autotoc_md365", null ]
      ] ],
      [ "Quick Start", "libpressio_python.html#autotoc_md367", null ],
      [ "from_config Structure", "libpressio_python.html#autotoc_md369", null ],
      [ "Encode and Decode", "libpressio_python.html#autotoc_md371", null ],
      [ "Pipeline Options", "libpressio_python.html#autotoc_md373", [
        [ "Error bound modes", "libpressio_python.html#autotoc_md374", null ],
        [ "Connections format", "libpressio_python.html#autotoc_md375", null ]
      ] ],
      [ "Stage Tokens", "libpressio_python.html#autotoc_md377", [
        [ "Lorenzo Predictor + Quantizer", "libpressio_python.html#autotoc_md378", null ],
        [ "Standalone Quantizer", "libpressio_python.html#autotoc_md379", null ],
        [ "Difference Stage", "libpressio_python.html#autotoc_md380", null ],
        [ "Zigzag and Negabinary Transforms", "libpressio_python.html#autotoc_md381", null ],
        [ "Run-Length Encoding (RLE)", "libpressio_python.html#autotoc_md382", null ],
        [ "Bitpacking", "libpressio_python.html#autotoc_md383", null ],
        [ "Bitshuffle", "libpressio_python.html#autotoc_md384", null ],
        [ "Repeated Zero Elimination (RZE)", "libpressio_python.html#autotoc_md385", null ],
        [ "Huffman Entropy Coding", "libpressio_python.html#autotoc_md386", null ]
      ] ],
      [ "Metrics", "libpressio_python.html#autotoc_md388", null ],
      [ "Common Recipes", "libpressio_python.html#autotoc_md390", [
        [ "Lorenzo + RLE (default)", "libpressio_python.html#autotoc_md391", null ],
        [ "Lorenzo + RZE (best ratio on smooth data)", "libpressio_python.html#autotoc_md392", null ],
        [ "Lorenzo + Bitshuffle", "libpressio_python.html#autotoc_md393", null ],
        [ "Quantizer with Inplace Outliers (float32 only)", "libpressio_python.html#autotoc_md394", null ],
        [ "Lossless Integer Lorenzo", "libpressio_python.html#autotoc_md395", null ],
        [ "3-D Structured Grid", "libpressio_python.html#autotoc_md396", null ]
      ] ],
      [ "CUDA Graph Mode", "libpressio_python.html#autotoc_md398", null ],
      [ "Exposing Stage Outputs", "libpressio_python.html#autotoc_md400", [
        [ "Stage output port names", "libpressio_python.html#autotoc_md401", null ]
      ] ],
      [ "TOML Config File", "libpressio_python.html#autotoc_md403", null ],
      [ "Error Handling", "libpressio_python.html#autotoc_md405", null ]
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
"annotated.html",
"classfz_1_1QuantizerStage.html#a295e54dc598074cca02587967d35b8e6",
"huffman__stage_8h.html#ad50d9a4b3988beaf4321dd448fc2f817a5999b8900bb8b90cfa1af137d355ff14",
"structfz_1_1FZMBufferEntry.html#a99571be38a6c75a0fabe34eb0c966e36"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';