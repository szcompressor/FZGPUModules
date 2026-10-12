// JIT context guards and concurrent first use of independent Auto pipelines.
#include "fzgpumodules.h"
#include "fused/common/nvrtc_jit.h"
#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct DeviceBuffer {
    void* ptr = nullptr;
    explicit DeviceBuffer(size_t bytes) { checkCuda(cudaMalloc(&ptr, bytes)); }
    ~DeviceBuffer() { cudaFree(ptr); }
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { checkCuda(cudaStreamCreate(&value)); }
    ~Stream() { cudaStreamDestroy(value); }
};
constexpr const char* kSource = "extern \"C\" __global__ void cache_probe() {}";
}

TEST(JitCache, CustomContextCannotBecomeCacheOwner) {
    checkCuda(cudaSetDevice(0));
    checkCuda(cudaFree(nullptr));
    CUcontext primary = nullptr;
    ASSERT_EQ(cuCtxGetCurrent(&primary), CUDA_SUCCESS);
    CUdevice device;
    ASSERT_EQ(cuCtxGetDevice(&device), CUDA_SUCCESS);
    CUcontext custom = nullptr;
    ASSERT_EQ(cuCtxCreate(&custom, 0, device), CUDA_SUCCESS);
    EXPECT_THROW(fz::fused::nvrtcGetKernel(kSource, "cache_probe"), std::runtime_error);
    EXPECT_EQ(cuCtxDestroy(custom), CUDA_SUCCESS);
    ASSERT_EQ(cuCtxSetCurrent(primary), CUDA_SUCCESS);
}

TEST(JitCache, ConcurrentFirstUseAndRepeatedSpecializedRoundTrip) {
    checkCuda(cudaSetDevice(0));
    checkCuda(cudaFree(nullptr));
    constexpr size_t n = 32768;
    constexpr size_t bytes = n * sizeof(float);
    constexpr int workers = 4;
    constexpr float eb = 1e-3f;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::string> errors(workers);
    std::vector<void*> kernels(workers, nullptr);
    std::vector<std::thread> threads;
    for (int slot = 0; slot < workers; ++slot) {
        threads.emplace_back([&, slot] {
            ++ready;
            while (!start.load()) std::this_thread::yield();
            try {
                checkCuda(cudaSetDevice(0));
                // Identical sources race on the same empty module-cache entry.
                kernels[slot] = fz::fused::nvrtcGetKernel(kSource, "cache_probe");
                Stream stream;
                DeviceBuffer input(bytes);
                std::vector<float> original(n);
                for (size_t i = 0; i < n; ++i)
                    original[i] = 0.5f * std::sin(i * 0.003f) + slot * 0.01f;
                checkCuda(cudaMemcpy(input.ptr, original.data(), bytes, cudaMemcpyHostToDevice));
                fz::Pipeline pipeline(bytes, fz::MemoryStrategy::PREALLOCATE);
                pipeline.setDims(n, 1, 1);
                pipeline.setFusionPolicy(fz::FusionPolicy::Auto);
                auto* q = pipeline.addStage<fz::QuantizerStage<float, uint32_t>>();
                q->setErrorBound(eb);
                q->setErrorBoundMode(fz::ErrorBoundMode::ABS);
                q->setLinearMode(true);
                auto* coder = pipeline.addStage<fz::AdaptiveBitpackStage<int32_t>>();
                coder->setBlockSize(32);
                coder->setOutlierSelection(false);
                pipeline.connect(coder, q, "codes");
                pipeline.finalize();
                if (pipeline.getFusedGroupCount() != 1)
                    throw std::runtime_error("Auto pipeline was not specialized");
                for (int iteration = 0; iteration < 5; ++iteration) {
                    void* compressed = nullptr;
                    void* decoded = nullptr;
                    size_t compressedBytes = 0, decodedBytes = 0;
                    pipeline.compress(input.ptr, bytes, &compressed, &compressedBytes, stream.value);
                    pipeline.decompress(compressed, compressedBytes, &decoded, &decodedBytes, stream.value);
                    checkCuda(cudaStreamSynchronize(stream.value));
                    if (decodedBytes != bytes) throw std::runtime_error("incorrect decoded length");
                    std::vector<float> result(n);
                    checkCuda(cudaMemcpy(result.data(), decoded, bytes, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < n; ++i)
                        if (!std::isfinite(result[i]) || std::abs(result[i] - original[i]) > eb * 1.001f)
                            throw std::runtime_error("specialized round-trip exceeds error bound");
                }
            } catch (const std::exception& error) {
                errors[slot] = error.what();
            }
        });
    }
    while (ready.load() != workers) std::this_thread::yield();
    start.store(true);
    for (auto& thread : threads) thread.join();
    for (int slot = 0; slot < workers; ++slot) {
        EXPECT_TRUE(errors[slot].empty()) << "worker " << slot << ": " << errors[slot];
        EXPECT_NE(kernels[slot], nullptr);
        EXPECT_EQ(kernels[slot], kernels[0]);
    }
}

TEST(JitCache, RejectCustomContextAndRestoreCachedPrimaryFunction) {
    checkCuda(cudaSetDevice(0));
    checkCuda(cudaFree(nullptr));
    CUcontext primary = nullptr;
    ASSERT_EQ(cuCtxGetCurrent(&primary), CUDA_SUCCESS);
    void* kernel = fz::fused::nvrtcGetKernel(kSource, "cache_probe");
    CUdevice device;
    ASSERT_EQ(cuCtxGetDevice(&device), CUDA_SUCCESS);
    CUcontext custom = nullptr;
    ASSERT_EQ(cuCtxCreate(&custom, 0, device), CUDA_SUCCESS);
    EXPECT_THROW(fz::fused::nvrtcGetKernel(kSource, "cache_probe"), std::runtime_error);
    // Guard applies to new sources as well as cache hits.
    EXPECT_THROW(fz::fused::nvrtcGetKernel(
        "extern \"C\" __global__ void uncached_probe() {}", "uncached_probe"), std::runtime_error);
    EXPECT_EQ(cuCtxDestroy(custom), CUDA_SUCCESS);
    ASSERT_EQ(cuCtxSetCurrent(primary), CUDA_SUCCESS);
    EXPECT_EQ(fz::fused::nvrtcGetKernel(kSource, "cache_probe"), kernel);
}
