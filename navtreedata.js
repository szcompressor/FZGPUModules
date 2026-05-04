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
    [ "Overview", "index.html#autotoc_md0", null ],
    [ "Requirements", "index.html#autotoc_md2", null ],
    [ "Quick Start", "index.html#autotoc_md4", null ],
    [ "Caller-Allocated Output (With Size Query)", "index.html#autotoc_md6", null ],
    [ "CUDA Graph Support", "index.html#autotoc_md8", null ],
    [ "Available Stages", "index.html#mainpage_stages", null ],
    [ "Memory Strategies", "index.html#autotoc_md11", null ],
    [ "File I/O", "index.html#autotoc_md13", null ],
    [ "Key API Classes", "index.html#autotoc_md15", null ],
    [ "Command Line Interface", "index.html#autotoc_md17", null ],
    [ "Building from Source", "index.html#autotoc_md19", null ],
    [ "Citation", "index.html#autotoc_md21", null ],
    [ "Stage Reference", "stages_overview.html", [
      [ "Predictors / Quantizers", "stages_overview.html#autotoc_md52", null ],
      [ "Coders", "stages_overview.html#autotoc_md53", null ],
      [ "Transforms / Shufflers", "stages_overview.html#autotoc_md54", null ],
      [ "LorenzoQuantStage", "stage_lorenzo_quant.html", [
        [ "What it does", "stage_lorenzo_quant.html#autotoc_md66", null ],
        [ "Template parameters", "stage_lorenzo_quant.html#autotoc_md68", null ],
        [ "Output ports (compression)", "stage_lorenzo_quant.html#autotoc_md70", null ],
        [ "Error bound modes", "stage_lorenzo_quant.html#autotoc_md72", null ],
        [ "Key setters", "stage_lorenzo_quant.html#autotoc_md74", null ],
        [ "Dimension setup — critical ordering rule", "stage_lorenzo_quant.html#autotoc_md76", null ],
        [ "CUDA Graph capture and NOA/REL modes", "stage_lorenzo_quant.html#autotoc_md78", null ],
        [ "Typical pipeline", "stage_lorenzo_quant.html#autotoc_md80", null ]
      ] ],
      [ "LorenzoStage", "stage_lorenzo.html", [
        [ "What it does", "stage_lorenzo.html#autotoc_md56", null ],
        [ "Template parameter", "stage_lorenzo.html#autotoc_md58", null ],
        [ "Ports", "stage_lorenzo.html#autotoc_md60", null ],
        [ "Dimension setup — critical ordering rule", "stage_lorenzo.html#autotoc_md62", null ],
        [ "Typical pipeline (cuSZp-style)", "stage_lorenzo.html#autotoc_md64", null ]
      ] ],
      [ "QuantizerStage", "stage_quantizer.html", [
        [ "What it does", "stage_quantizer.html#autotoc_md82", null ],
        [ "Template parameters", "stage_quantizer.html#autotoc_md84", null ],
        [ "Output ports (compression)", "stage_quantizer.html#autotoc_md86", [
          [ "Normal mode (4 outputs)", "stage_quantizer.html#autotoc_md87", null ],
          [ "Inplace outlier mode (1 output)", "stage_quantizer.html#autotoc_md88", null ]
        ] ],
        [ "Error bound modes", "stage_quantizer.html#autotoc_md90", null ],
        [ "Key setters", "stage_quantizer.html#autotoc_md92", null ],
        [ "Inplace outlier mode constraints", "stage_quantizer.html#autotoc_md94", [
          [ "1. Zigzag encoding must be enabled", "stage_quantizer.html#autotoc_md95", null ],
          [ "2. <tt>sizeof(TCode) == sizeof(TInput)</tt>", "stage_quantizer.html#autotoc_md96", null ]
        ] ],
        [ "CUDA Graph capture and NOA/REL modes", "stage_quantizer.html#autotoc_md98", null ],
        [ "Typical pipelines", "stage_quantizer.html#autotoc_md100", [
          [ "PFPL-style (standalone quantizer)", "stage_quantizer.html#autotoc_md101", null ],
          [ "Inplace outlier mode", "stage_quantizer.html#autotoc_md102", null ]
        ] ]
      ] ],
      [ "RLEStage", "stage_rle.html", [
        [ "What it does", "stage_rle.html#autotoc_md104", null ],
        [ "Template parameter", "stage_rle.html#autotoc_md106", null ],
        [ "Wire format", "stage_rle.html#autotoc_md108", null ],
        [ "No chunk size setter", "stage_rle.html#autotoc_md110", null ],
        [ "Typical pipeline", "stage_rle.html#autotoc_md112", null ]
      ] ],
      [ "RZEStage", "stage_rze.html", [
        [ "What it does", "stage_rze.html#autotoc_md114", null ],
        [ "Output stream layout", "stage_rze.html#autotoc_md116", null ],
        [ "Key setters", "stage_rze.html#autotoc_md118", null ],
        [ "CUDA Graph compatibility", "stage_rze.html#autotoc_md120", null ],
        [ "Alignment requirement", "stage_rze.html#autotoc_md122", null ],
        [ "Typical pipeline", "stage_rze.html#autotoc_md124", null ]
      ] ],
      [ "BitpackStage", "stage_bitpack.html", [
        [ "What it does", "stage_bitpack.html#autotoc_md23", null ],
        [ "Template parameter", "stage_bitpack.html#autotoc_md25", null ],
        [ "Key setter", "stage_bitpack.html#autotoc_md27", null ],
        [ "CUDA Graph compatibility", "stage_bitpack.html#autotoc_md29", null ],
        [ "Typical pipeline (cuSZp-style)", "stage_bitpack.html#autotoc_md31", null ]
      ] ],
      [ "DifferenceStage", "stage_diff.html", [
        [ "What it does", "stage_diff.html#autotoc_md41", null ],
        [ "Template parameters", "stage_diff.html#autotoc_md43", null ],
        [ "Chunking", "stage_diff.html#autotoc_md45", null ],
        [ "Key setters", "stage_diff.html#autotoc_md47", null ],
        [ "Common instantiations", "stage_diff.html#autotoc_md49", null ],
        [ "Typical pipeline", "stage_diff.html#autotoc_md51", null ]
      ] ],
      [ "BitshuffleStage", "stage_bitshuffle.html", [
        [ "What it does", "stage_bitshuffle.html#autotoc_md33", null ],
        [ "Key setters", "stage_bitshuffle.html#autotoc_md35", null ],
        [ "Alignment requirement", "stage_bitshuffle.html#autotoc_md37", null ],
        [ "Typical pipeline", "stage_bitshuffle.html#autotoc_md39", null ]
      ] ]
    ] ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", null ],
        [ "Functions", "namespacemembers_func.html", null ],
        [ "Variables", "namespacemembers_vars.html", null ],
        [ "Enumerations", "namespacemembers_enum.html", null ]
      ] ]
    ] ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"annotated.html",
"classfz_1_1NegabinaryStage.html#ad835f98c81bdf913e0aa2757d0c02823",
"dir_27d071c2886d8c6831e2265dab8e4463.html",
"stages_overview.html#autotoc_md53",
"structfz_1_1QuantizerStage_1_1Config.html#a78c5165df664c645904368bfe14e3dbc"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';