#!/usr/bin/env bash
# new_stage.sh — scaffold a new FZGPUModules stage
#
# Usage:
#   scripts/new_stage.sh <StageName> <category> [StageTypeId]
#
#   StageName   : PascalCase name, e.g. "MyEncoder"
#   category    : "transforms", "encoders", or "predictors"
#   StageTypeId : integer for the StageType enum (optional — prints a reminder if omitted)
#
# Example:
#   scripts/new_stage.sh MyEncoder encoders 19
#
# What it creates:
#   modules/<category>/<lower>/
#     <lower>_stage.h
#     <lower>_stage.cu
#   tests/stages/test_<lower>.cpp
#
# What it prints (must be done manually):
#   - StageType enum entry in include/fzm_format.h
#   - stageTypeToString() case in include/fzm_format.h
#   - createStage() case in include/stage/stage_factory.h
#   - CMakeLists.txt library addition

set -euo pipefail

# ── Args ───────────────────────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <StageName> <category> [StageTypeId]"
    echo "  category: transforms | encoders | predictors"
    exit 1
fi

NAME="$1"       # e.g. MyEncoder (without "Stage" suffix)
CATEGORY="$2"   # transforms | encoders | predictors
TYPE_ID="${3:-}"

# Strip trailing "Stage" if the user included it, so class is always <NAME>Stage
NAME="${NAME%Stage}"

LOWER=$(echo "$NAME" | sed 's/\([A-Z]\)/_\1/g' | sed 's/^_//' | tr '[:upper:]' '[:lower:]')
# e.g. MyEncoder → my_encoder

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODULE_DIR="${REPO_ROOT}/modules/${CATEGORY}/${LOWER}"
TEST_FILE="${REPO_ROOT}/tests/stages/test_${LOWER}.cpp"

# ── Sanity checks ──────────────────────────────────────────────────────────────
if [[ "$CATEGORY" != "transforms" && "$CATEGORY" != "encoders" && "$CATEGORY" != "predictors" ]]; then
    echo "Error: category must be one of: transforms, encoders, predictors"
    exit 1
fi

if [[ -d "$MODULE_DIR" ]]; then
    echo "Error: directory already exists: $MODULE_DIR"
    exit 1
fi

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
        // TODO: return a safe upper bound. This assumes size-preserving.
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
        // TODO: replace with the correct StageType enum value after adding it
        // to include/fzm_format.h.
        return static_cast<uint16_t>(StageType::UNKNOWN);
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
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#include "${CATEGORY}/${LOWER}/${LOWER}_stage.h"
#include "helpers/stage_harness.h"
#include "helpers/fz_test_utils.h"
#include "pipeline/compressor.h"
#include "mem/mempool.h"

// TODO: adjust T and TOut to the types your stage uses
using T    = float;
using TOut = float;

// ── ForwardRoundTrip ──────────────────────────────────────────────────────────
// Forward compression followed by inverse decompression produces correct output.
TEST(${NAME}Stage, ForwardRoundTrip) {
    const size_t n = 1024;
    auto data = fz_test::make_smooth_data<T>(n);

    fz::Pipeline p(fz::MemoryPoolConfig(n * sizeof(T)));
    auto* stage = p.addStage<fz::${NAME}Stage>();
    // TODO: connect stages if needed
    p.finalize();

    // Upload
    T* d_in = nullptr;
    cudaMalloc(&d_in, n * sizeof(T));
    cudaMemcpy(d_in, data.data(), n * sizeof(T), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in, n * sizeof(T), &d_comp, &comp_sz, stream);

    void* d_out = nullptr; size_t out_sz = 0;
    p.decompress(d_comp, comp_sz, &d_out, &out_sz, stream);
    cudaStreamSynchronize(stream);

    std::vector<T> result(n);
    cudaMemcpy(result.data(), d_out, out_sz, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);

    // TODO: adjust tolerance or use EXPECT_EQ for lossless stages
    for (size_t i = 0; i < n; i++)
        EXPECT_NEAR(result[i], data[i], 1e-4f) << "mismatch at i=" << i;
}

// ── ZeroInput ─────────────────────────────────────────────────────────────────
TEST(${NAME}Stage, ZeroInput) {
    fz::Pipeline p(fz::MemoryPoolConfig(0));
    p.addStage<fz::${NAME}Stage>();
    p.finalize();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    T dummy = 0;
    T* d_in = nullptr;
    cudaMalloc(&d_in, sizeof(T));
    cudaMemcpy(d_in, &dummy, sizeof(T), cudaMemcpyHostToDevice);

    void* d_comp = nullptr; size_t comp_sz = 0;
    // Zero-size compress should not crash
    EXPECT_NO_THROW(p.compress(d_in, 0, &d_comp, &comp_sz, stream));
    cudaStreamSynchronize(stream);
    cudaFree(d_in);
    cudaStreamDestroy(stream);
}

// ── SerializeDeserialize ──────────────────────────────────────────────────────
TEST(${NAME}Stage, SerializeDeserialize) {
    fz::${NAME}Stage original;
    uint8_t buf[128] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));

    fz::${NAME}Stage restored;
    restored.deserializeHeader(buf, written);

    // TODO: EXPECT_EQ the relevant config fields between original and restored
    // e.g. EXPECT_EQ(original.getConfig(), restored.getConfig());
    SUCCEED(); // replace with real assertions
}

// ── PipelineIntegration ───────────────────────────────────────────────────────
TEST(${NAME}Stage, PipelineIntegration) {
    const size_t n = 4096;
    auto data = fz_test::make_smooth_data<T>(n);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    T* d_in = nullptr;
    cudaMalloc(&d_in, n * sizeof(T));
    cudaMemcpy(d_in, data.data(), n * sizeof(T), cudaMemcpyHostToDevice);

    fz::Pipeline p(fz::MemoryPoolConfig(n * sizeof(T)));
    p.addStage<fz::${NAME}Stage>();
    p.finalize();

    void* d_comp = nullptr; size_t comp_sz = 0;
    ASSERT_NO_THROW(p.compress(d_in, n * sizeof(T), &d_comp, &comp_sz, stream));
    EXPECT_GT(comp_sz, 0u);

    void* d_out = nullptr; size_t out_sz = 0;
    ASSERT_NO_THROW(p.decompress(d_comp, comp_sz, &d_out, &out_sz, stream));
    cudaStreamSynchronize(stream);

    std::vector<T> result(n);
    cudaMemcpy(result.data(), d_out, out_sz, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);

    EXPECT_EQ(out_sz, n * sizeof(T));
}

// ── GraphCompatible ───────────────────────────────────────────────────────────
TEST(${NAME}Stage, GraphCompatible) {
    fz::${NAME}Stage stage;
    // TODO: change expected value to false if execute() does D2H transfers
    EXPECT_TRUE(stage.isGraphCompatible());
}
TESTS

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "✓ Created:"
echo "    ${MODULE_DIR}/${LOWER}_stage.h"
echo "    ${MODULE_DIR}/${LOWER}_stage.cu"
echo "    ${TEST_FILE}"
echo ""
echo "Manual steps remaining:"
echo ""

if [[ -n "$TYPE_ID" ]]; then
    echo "  1. include/fzm_format.h — add to StageType enum:"
    echo "       ${NAME^^} = ${TYPE_ID},"
else
    echo "  1. include/fzm_format.h — add to StageType enum:"
    echo "       ${NAME^^} = <next_available_id>,  // check existing values first"
fi

echo ""
echo "  2. include/fzm_format.h — add to stageTypeToString():"
echo "       case StageType::${NAME^^}: return \"${NAME}\";"
echo ""
echo "  3. include/stage/stage_factory.h — add to createStage():"
echo "       case StageType::${NAME^^}: {"
echo "           stage = new ${NAME}Stage();"
echo "           if (config_size > 0) stage->deserializeHeader(config, config_size);"
echo "           break;"
echo "       }"
echo ""
echo "  4. CMakeLists.txt — add .cu to the library target:"
echo "       modules/${CATEGORY}/${LOWER}/${LOWER}_stage.cu"
echo ""
echo "  5. tests/stages/CMakeLists.txt — register the test:"
echo "       fz_add_test(test_${LOWER} test_${LOWER}.cpp LABELS stages gpu)"
echo ""
echo "See memory/how_to_add_a_stage.md for full details."
