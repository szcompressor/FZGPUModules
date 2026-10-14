// Multi-source archive/header decoding with a selected second Quantizer source.
#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"
#include <cstdio>
#include <string>
#include <vector>
using namespace fz;
using namespace fz_test;

TEST(MultiSourcePipeline, PartialChunkArchiveAndPrimedSecondSource) {
    {
    CudaStream stream;
    constexpr size_t count = 16387;
    constexpr float bound = 1e-3f;
    const size_t bytes = count * sizeof(float);
    auto input = make_smooth_data<float>(count);
    CudaBuffer<float> device_input(count);
    device_input.upload(input, stream);
    auto build = [&](Pipeline& p) {
        p.setSpecializationPolicy(SpecializationPolicy::Off);
        for (int source = 0; source < 2; ++source) {
            auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
            q->setErrorBound(source == 0 ? 1e-1f : bound);
            q->setErrorBoundMode(ErrorBoundMode::ABS);
            q->setLinearMode(true);
            if (source == 0) {
                auto* d = p.addStage<DifferenceStage<uint32_t>>();
                d->setChunkSize(16384);
                p.connect(d, q, "codes");
                auto* b = p.addStage<BitshuffleStage>();
                b->setElementWidth(4);
                b->setBlockSize(16384);
                p.connect(b, d);
            } else {
                auto* d = p.addStage<DifferenceStage<uint32_t>>();
                d->setChunkSize(16384);
                p.connect(d, q, "codes");
                p.setPrimarySource(q);
            }
        }
        p.finalize();
    };
    Pipeline producer(bytes, MemoryStrategy::PREALLOCATE, 8.0f);
    Pipeline consumer(bytes, MemoryStrategy::PREALLOCATE, 8.0f);
    build(producer);
    build(consumer);
    void* compressed = nullptr;
    size_t compressed_bytes = 0;
    producer.compress(device_input.void_ptr(), bytes, &compressed, &compressed_bytes, stream);
    auto header = producer.buildHeader();
    ASSERT_EQ(header.core.num_sources, 2u);
    EXPECT_EQ(header.core.uncompressed_size, bytes);
    for (size_t i = 0; i < 2; ++i) EXPECT_EQ(header.core.source_uncompressed_sizes[i], bytes);
    const std::string path = "/tmp/fzgmod_multi_partial.fzm";
    producer.writeToFile(path, stream);
    void* decoded = nullptr;
    size_t decoded_bytes = 0;
    Pipeline::decompressFromFile(path, &decoded, &decoded_bytes, stream);
    ASSERT_EQ(decoded_bytes, bytes);
    std::vector<float> file_output(count);
    FZ_TEST_CUDA(cudaMemcpy(file_output.data(), decoded, bytes, cudaMemcpyDeviceToHost));
    FZ_TEST_CUDA(cudaFree(decoded));
    EXPECT_LE(max_abs_error(input, file_output), bound * 1.01f);
    std::remove(path.c_str());
    auto serialized = producer.serializeHeaderToMemory();
    consumer.primeInverseFromHeader(serialized.data(), serialized.size());
    consumer.decompress(compressed, compressed_bytes, &decoded, &decoded_bytes, stream);
    ASSERT_EQ(decoded_bytes, bytes);
    std::vector<float> memory_output(count);
    FZ_TEST_CUDA(cudaMemcpy(memory_output.data(), decoded, bytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(file_output, memory_output);
    auto fresh = Pipeline::fromHeader(serialized.data(), serialized.size());
    fresh->setPoolManagedDecompOutput(false);
    fresh->decompress(compressed, compressed_bytes, &decoded, &decoded_bytes, stream);
    stream.sync();
    ASSERT_EQ(decoded_bytes, bytes);
    std::vector<float> fresh_output(count);
    FZ_TEST_CUDA(cudaMemcpy(fresh_output.data(), decoded, bytes, cudaMemcpyDeviceToHost));
    FZ_TEST_CUDA(cudaFree(decoded));
    EXPECT_EQ(file_output, fresh_output);
    }
    FZ_TEST_CUDA(cudaDeviceSynchronize());
}
