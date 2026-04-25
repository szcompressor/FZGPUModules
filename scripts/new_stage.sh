#!/usr/bin/env bash
# new_stage.sh — scaffold a new FZGPUModules stage
#
# Usage:
#   scripts/new_stage.sh <StageName> <category> [StageTypeId]
#
#   StageName   : PascalCase name, e.g. "MyEncoder"
#   category    : "transforms", "encoders", or "predictors"
#   StageTypeId : integer for the StageType enum (optional — auto-detected if omitted)
#
# Examples:
#   scripts/new_stage.sh MyEncoder encoders       # ID auto-assigned
#   scripts/new_stage.sh MyEncoder encoders 42    # ID explicitly pinned
#
# What it creates/modifies automatically:
#   modules/<category>/<lower>/
#     <lower>_stage.h
#     <lower>_stage.cu
#   tests/stages/test_<lower>.cpp
#   include/fzm_format.h         — StageType enum entry + stageTypeToString() case
#   CMakeLists.txt               — .cu added to fzgmod_encoders/predictors/transforms target
#   tests/stages/CMakeLists.txt  — fz_add_test() entry
#
# What still needs manual attention (printed at the end):
#   include/stage/stage_factory.h  — createStage() case
#   src/pipeline/config.cpp        — TOML load/save

set -euo pipefail

# ── Args ───────────────────────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <StageName> <category> [StageTypeId]"
    echo "  category: transforms | encoders | predictors"
    echo "  StageTypeId: optional integer — lowest unused value is auto-selected if omitted"
    exit 1
fi

NAME="$1"       # e.g. MyEncoder (without "Stage" suffix)
CATEGORY="$2"   # transforms | encoders | predictors

# Strip trailing "Stage" if the user included it, so class is always <NAME>Stage
NAME="${NAME%Stage}"

LOWER=$(echo "$NAME" | sed 's/\([A-Z]\)/_\1/g' | sed 's/^_//' | tr '[:upper:]' '[:lower:]')
UPPER=$(echo "$NAME" | sed 's/\([A-Z]\)/_\1/g' | sed 's/^_//' | tr '[:lower:]' '[:upper:]')

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FZM_FORMAT_H="${REPO_ROOT}/include/fzm_format.h"

# ── StageTypeId: explicit or auto-detect lowest unused ────────────────────────
if [[ $# -ge 3 ]]; then
    TYPE_ID="$3"
    if ! [[ "$TYPE_ID" =~ ^[0-9]+$ ]]; then
        echo "Error: StageTypeId must be a non-negative integer, got: $TYPE_ID"
        exit 1
    fi
    echo "  Using explicit StageTypeId: ${TYPE_ID}"
else
    # Find the lowest positive integer not already assigned in the StageType enum.
    # Reads all "= <number>," occurrences from the enum block, sorts them, then
    # walks from 1 upward to find the first gap.
    TYPE_ID=$(grep -oP '=\s*\K[0-9]+(?=\s*,)' "$FZM_FORMAT_H" \
        | sort -n \
        | awk 'BEGIN{e=1} {if($1==e)e++} END{print e}')
    echo "  Auto-selected StageTypeId: ${TYPE_ID} (lowest unused value)"
fi

MODULE_DIR="${REPO_ROOT}/modules/${CATEGORY}/${LOWER}"
TEST_FILE="${REPO_ROOT}/tests/stages/test_${LOWER}.cpp"
FZM_FORMAT_H="${REPO_ROOT}/include/fzm_format.h"
CMAKELISTS="${REPO_ROOT}/CMakeLists.txt"
TESTS_CMAKELISTS="${REPO_ROOT}/tests/stages/CMakeLists.txt"

# ── Sanity checks ──────────────────────────────────────────────────────────────
if [[ "$CATEGORY" != "transforms" && "$CATEGORY" != "encoders" && "$CATEGORY" != "predictors" ]]; then
    echo "Error: category must be one of: transforms, encoders, predictors"
    exit 1
fi

if [[ -d "$MODULE_DIR" ]]; then
    echo "Error: directory already exists: $MODULE_DIR"
    exit 1
fi

# Check StageTypeId is not already in use
if grep -q "= ${TYPE_ID}," "$FZM_FORMAT_H" || grep -q "= ${TYPE_ID}$" "$FZM_FORMAT_H"; then
    echo "Error: StageTypeId ${TYPE_ID} is already used in include/fzm_format.h"
    echo "  Check existing values and pick a different ID."
    exit 1
fi

# ── Determine which CMake library target to add the .cu to ────────────────────
# All three categories currently go into fzgmod_encoders; adjust if that changes.
CMAKE_TARGET="fzgmod_encoders"

# ── Create module directory ────────────────────────────────────────────────────
mkdir -p "$MODULE_DIR"

# ── Header ────────────────────────────────────────────────────────────────────
cat > "${MODULE_DIR}/${LOWER}_stage.h" << HEADER
#pragma once

/**
 * @file ${LOWER}_stage.h
 * @brief ${NAME} stage — TODO: one-line description.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

/**
 * ${NAME} stage.
 *
 * TODO: describe what this stage does, its input/output types,
 * and any configuration parameters.
 */
class ${NAME}Stage : public Stage {
public:
    ${NAME}Stage() = default;

    // ── Stage control ─────────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    // ── Execution ─────────────────────────────────────────────────────────────
    void execute(
        cudaStream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    // ── Metadata ──────────────────────────────────────────────────────────────
    std::string getName() const override { return "${NAME}"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        // TODO: return a safe upper bound for both forward AND inverse directions.
        // Non-size-preserving stages must check is_inverse_ and return the correct
        // bound for each direction — the DAG allocates output buffers before execute().
        return {input_sizes.empty() ? 0 : input_sizes[0]};
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }

    size_t getActualOutputSize(int index) const override {
        return (index == 0) ? actual_output_size_ : 0;
    }

    // ── Type system ───────────────────────────────────────────────────────────
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::${UPPER});
    }

    uint8_t getOutputDataType(size_t /*output_index*/) const override {
        // TODO: return the DataType of the output, or DataType::UNKNOWN
        // for byte-transparent stages.
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }

    uint8_t getInputDataType(size_t /*input_index*/) const override {
        // TODO: return the expected DataType of the input, or DataType::UNKNOWN
        // to opt out of finalize() type-checking.
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }

    // ── Serialization ─────────────────────────────────────────────────────────
    size_t serializeHeader(
        size_t output_index, uint8_t* buf, size_t max_size
    ) const override {
        // TODO: write config bytes into buf (max 128 bytes). Return bytes written.
        (void)output_index; (void)buf; (void)max_size;
        return 0;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        // TODO: restore config from buf.
        (void)buf; (void)size;
    }

    size_t getMaxHeaderSize(size_t /*output_index*/) const override {
        return 0; // TODO: update to match bytes written in serializeHeader()
    }

    // Uncomment if deserializeHeader() overwrites fields also used by forward passes:
    // void saveState()    override { saved_config_ = config_; }
    // void restoreState() override { config_ = saved_config_; }

    // Uncomment if this stage holds persistent pool allocations:
    // size_t estimateScratchBytes(const std::vector<size_t>& input_sizes) const override;

    // Uncomment if execute() does D2H copies or host-side branching on device data:
    // bool isGraphCompatible() const override { return false; }

    // Uncomment if input must be aligned to a chunk boundary:
    // size_t getRequiredInputAlignment() const override { return chunk_bytes_; }

private:
    bool   is_inverse_         = false;
    size_t actual_output_size_ = 0;
    // TODO: add config fields here
};

} // namespace fz
HEADER

# ── Implementation ────────────────────────────────────────────────────────────
cat > "${MODULE_DIR}/${LOWER}_stage.cu" << IMPL
#include "${CATEGORY}/${LOWER}/${LOWER}_stage.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace fz {

// ── Device kernels ─────────────────────────────────────────────────────────────
// TODO: add __global__ kernel(s) here.
// All kernels must be launched on the provided stream — never call
// cudaDeviceSynchronize() inside execute().

// ── ${NAME}Stage::execute ──────────────────────────────────────────────────────
void ${NAME}Stage::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    (void)pool; // remove if you use pool->allocate()

    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error(
            "${NAME}Stage: inputs, outputs, and sizes must be non-empty");

    const size_t byte_size = sizes[0];
    if (byte_size == 0) {
        actual_output_size_ = 0;
        return;
    }

    // TODO: launch kernel(s). Example pattern:
    //   constexpr int kBlock = 256;
    //   const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    //   myKernel<<<grid, kBlock, 0, stream>>>(...);
    //   FZ_CUDA_CHECK(cudaGetLastError());

    actual_output_size_ = byte_size; // TODO: set to actual output bytes written
}

} // namespace fz
IMPL

# ── Test skeleton ─────────────────────────────────────────────────────────────
cat > "$TEST_FILE" << TESTS
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "${CATEGORY}/${LOWER}/${LOWER}_stage.h"
#include "helpers/stage_harness.h"
#include "helpers/fz_test_utils.h"

using namespace fz;
using namespace fz_test;

// ── RoundTrip ────────────────────────────────────────────────────────────────
// Forward compression followed by inverse decompression produces correct output.
// Uses a single Pipeline instance — compress() and decompress() must share state.
TEST(${NAME}Stage, RoundTrip) {
    const size_t N = 4096;
    const size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth_data<float>(N);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<${NAME}Stage>();
    // TODO: set stage parameters and connect if not the first stage
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    // TODO: adjust tolerance or use EXPECT_EQ for lossless stages
    EXPECT_LT(res.max_error, 1e-4f);
}

// ── ZeroInput ─────────────────────────────────────────────────────────────────
TEST(${NAME}Stage, ZeroInput) {
    Pipeline p(0, MemoryStrategy::PREALLOCATE);
    p.addStage<${NAME}Stage>();
    p.finalize();

    CudaStream cs;
    std::vector<float> empty;
    EXPECT_NO_THROW({
        auto res = pipeline_round_trip<float>(p, empty, cs.stream);
    });
}

// ── SerializeDeserialize ──────────────────────────────────────────────────────
TEST(${NAME}Stage, SerializeDeserialize) {
    ${NAME}Stage original;
    // TODO: set parameters on original

    uint8_t buf[128] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));

    ${NAME}Stage restored;
    restored.deserializeHeader(buf, written);

    // TODO: EXPECT_EQ the relevant config fields between original and restored
    // e.g. EXPECT_EQ(original.getFoo(), restored.getFoo());
    SUCCEED(); // replace with real assertions
}

// ── SaveRestoreState ──────────────────────────────────────────────────────────
// Only needed if deserializeHeader() overwrites fields used by forward passes.
// The pipeline calls saveState() before and restoreState() after each decompress.
TEST(${NAME}Stage, SaveRestoreState) {
    ${NAME}Stage s;
    // TODO: set a parameter, saveState, call deserializeHeader with different
    // bytes, restoreState, verify the original value is back.
    SUCCEED(); // replace with real assertions
}

// ── PipelineIntegration ───────────────────────────────────────────────────────
// Wires the stage into a full pipeline and verifies end-to-end round-trip.
// IMPORTANT: use one Pipeline instance — decompress() builds the inverse DAG
// from the same object that ran compress(). Two separate pipelines will throw.
TEST(${NAME}Stage, PipelineIntegration) {
    const size_t N = 4096;
    const size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth_data<float>(N);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.addStage<${NAME}Stage>();
    // TODO: add other stages, connect ports
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LT(res.max_error, 1e-4f);
    // TODO: EXPECT_LT(res.compressed_bytes, in_bytes) if stage is compressive
}

// ── GraphCompatible ───────────────────────────────────────────────────────────
TEST(${NAME}Stage, GraphCompatible) {
    ${NAME}Stage stage;
    // TODO: change expected value to false if execute() does D2H transfers
    EXPECT_TRUE(stage.isGraphCompatible());
}
TESTS

# ── fzm_format.h: StageType enum entry ───────────────────────────────────────
# Insert before the closing brace of the enum (the line with just "};")
# We target the last enum value line (ends with a comment) and insert after it.
# Strategy: find the RZE line (currently last) and insert after it.
LAST_ENUM_LINE="    RZE        = 18,"
NEW_ENUM_LINE="    ${UPPER}    = ${TYPE_ID},"
if grep -q "${UPPER}" "$FZM_FORMAT_H"; then
    echo "Warning: ${UPPER} already appears in fzm_format.h — skipping enum insertion."
else
    # Find the last "= <number>," line in the StageType enum and append after it.
    # We use Python for reliable multi-line in-place editing.
    python3 - "$FZM_FORMAT_H" "$UPPER" "$TYPE_ID" << 'PYEOF'
import sys, re

path   = sys.argv[1]
upper  = sys.argv[2]
tid    = sys.argv[3]

text = open(path).read()

# Match the last "    IDENTIFIER = NUMBER,  ///< ..." line inside the enum
# and insert the new entry immediately after it.
pattern = r'([ \t]+\w+\s*=\s*\d+,\s*///[^\n]*\n)(\s*\};)'
new_entry = f'    {upper:<10} = {tid},   ///< TODO: describe this stage\n'
replacement = r'\1' + new_entry + r'\2'

new_text, n = re.subn(pattern, replacement, text, count=1)
if n == 0:
    print(f"Warning: could not find insertion point in StageType enum — edit fzm_format.h manually.")
    sys.exit(0)

open(path, 'w').write(new_text)
print(f"  + StageType::{upper} = {tid} added to enum")
PYEOF
fi

# ── fzm_format.h: stageTypeToString() case ────────────────────────────────────
if grep -q "StageType::${UPPER}" "$FZM_FORMAT_H"; then
    echo "  (stageTypeToString case already present — skipping)"
else
    python3 - "$FZM_FORMAT_H" "$UPPER" "$NAME" << 'PYEOF'
import sys, re

path  = sys.argv[1]
upper = sys.argv[2]
name  = sys.argv[3]

text = open(path).read()

# Insert before the "default:" line in stageTypeToString
new_case = f'        case StageType::{upper}:  return "{name}";\n'
new_text = text.replace(
    '        default:                     return "Unknown";',
    new_case + '        default:                     return "Unknown";',
    1
)
if new_text == text:
    print("Warning: could not patch stageTypeToString() — add case manually.")
else:
    open(path, 'w').write(new_text)
    print(f"  + stageTypeToString() case for {upper} added")
PYEOF
fi

# ── CMakeLists.txt: add .cu to library target ─────────────────────────────────
CU_ENTRY="  modules/${CATEGORY}/${LOWER}/${LOWER}_stage.cu"
if grep -qF "modules/${CATEGORY}/${LOWER}/${LOWER}_stage.cu" "$CMAKELISTS"; then
    echo "  (CMakeLists.txt entry already present — skipping)"
else
    python3 - "$CMAKELISTS" "$CU_ENTRY" "$CMAKE_TARGET" << 'PYEOF'
import sys

path      = sys.argv[1]
cu_entry  = sys.argv[2]
target    = sys.argv[3]

text = open(path).read()

# Find the add_library(fzgmod_encoders ...) block and insert the .cu before the closing paren.
# We look for the last existing modules/ line inside that block and insert after it.
import re

# Match the block: add_library(<target>\n  ...sources...\n)
block_pattern = rf'(add_library\({re.escape(target)}\n(?:  [^\n]+\n)*?)(  modules/[^\n]+\n)(\))'
def inserter(m):
    return m.group(1) + m.group(2) + cu_entry + '\n' + m.group(3)

new_text, n = re.subn(block_pattern, inserter, text, count=1)
if n == 0:
    print(f"Warning: could not find {target} add_library block — add .cu manually.")
else:
    open(path, 'w').write(new_text)
    print(f"  + {cu_entry.strip()} added to {target}")
PYEOF
fi

# ── tests/stages/CMakeLists.txt: fz_add_test entry ───────────────────────────
if grep -q "test_${LOWER}" "$TESTS_CMAKELISTS"; then
    echo "  (tests/stages/CMakeLists.txt entry already present — skipping)"
else
    cat >> "$TESTS_CMAKELISTS" << CMTEST

# ─── ${NAME} stage ────────────────────────────────────────────────────────────
fz_add_test(
    NAME    test_${LOWER}
    SOURCES test_${LOWER}.cpp
    LIBS    fzgmod_encoders fzgmod_mem fzgmod
    LABELS  stages gpu
)
CMTEST
    echo "  + test_${LOWER} added to tests/stages/CMakeLists.txt"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Done. Created/modified:"
echo "    ${MODULE_DIR}/${LOWER}_stage.h"
echo "    ${MODULE_DIR}/${LOWER}_stage.cu"
echo "    ${TEST_FILE}"
echo "    ${FZM_FORMAT_H}  (StageType enum + stageTypeToString)"
echo "    ${CMAKELISTS}    (.cu added to ${CMAKE_TARGET})"
echo "    ${TESTS_CMAKELISTS}  (fz_add_test entry)"
echo ""
echo "Two steps still require manual edits:"
echo ""
echo "  1. include/stage/stage_factory.h — add createStage() case:"
echo "       case StageType::${UPPER}: {"
echo "           auto* s = new ${NAME}Stage();"
echo "           if (config_size > 0) s->deserializeHeader(config, config_size);"
echo "           stage = s;"
echo "           break;"
echo "       }"
echo ""
echo "  2. src/pipeline/config.cpp — add TOML load/save support:"
echo "       #include \"${CATEGORY}/${LOWER}/${LOWER}_stage.h\""
echo "       static Stage* add${NAME}Stage(Pipeline&, const toml::table&) { ... }"
echo "       } else if (type == \"${NAME}\") { s = add${NAME}Stage(*this, *t); }"
echo "       // + saveConfig() case"
echo ""
echo "See memory/how_to_add_a_stage.md for full details."
