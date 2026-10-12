// External consumer: uses only installed public headers/targets.
#include <fzgpumodules.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fz;

namespace {
constexpr size_t kCount = 32768; // aligned archive reload
constexpr size_t kPartialCount = 32771; // partial warp block and partial chunk
constexpr float kBound = 1e-3f;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct Allocation {
    void* ptr = nullptr;
    explicit Allocation(size_t bytes) { check(cudaMalloc(&ptr, bytes)); }
    ~Allocation() { if (ptr) cudaFree(ptr); }
};

void build(Pipeline& p, bool chunk, size_t count) {
    p.setDims(count, 1, 1);
    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(kBound);
    q->setErrorBoundMode(ErrorBoundMode::ABS);
    if (chunk) {
        q->setQuantRadius(1 << 22);
        q->setZigzagCodes(true);
        q->setInplaceOutliers(true);
        auto* d = p.addStage<DifferenceStage<int32_t, uint32_t>>();
        d->setChunkSize(16384);
        p.connect(d, q, "codes");
        auto* b = p.addStage<BitshuffleStage>();
        b->setElementWidth(4); b->setBlockSize(16384);
        p.connect(b, d);
        auto* r = p.addStage<RZEStage>();
        r->setWordSize(1); r->setChunkSize(16384);
        p.connect(r, b);
    } else {
        q->setLinearMode(true);
        auto* l = p.addStage<LorenzoStage<int32_t>>();
        l->setBlockSize(32);
        p.connect(l, q, "codes");
        auto* a = p.addStage<AdaptiveBitpackStage<int32_t>>();
        a->setBlockSize(32); a->setOutlierSelection(true);
        p.connect(a, l);
    }
}

std::vector<float> copyReconstruction(const void* ptr, size_t bytes, size_t count) {
    require(bytes == count * sizeof(float), "wrong reconstruction size");
    std::vector<float> out(count);
    check(cudaMemcpy(out.data(), ptr, bytes, cudaMemcpyDeviceToHost));
    return out;
}

void write(const std::string& directory, bool chunk, size_t count) {
    const std::string stem = directory + (chunk ? "/chunk" : "/warp") + std::to_string(count);
    const size_t bytes = count * sizeof(float);
    std::vector<float> input(count);
    for (size_t i = 0; i < count; ++i)
        input[i] = 0.5f * std::sin(i * 0.001f) + 0.2f * std::cos(i * 0.017f);
    Allocation deviceInput(bytes);
    check(cudaMemcpy(deviceInput.ptr, input.data(), bytes, cudaMemcpyHostToDevice));
    std::vector<unsigned char> stagedPayload;
    std::vector<float> stagedOutput;
    for (bool specialize : {false, true}) {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE);
        p.setSpecializationPolicy(specialize ? SpecializationPolicy::Auto
                                            : SpecializationPolicy::Off);
        build(p, chunk, count);
        p.finalize();
        require(p.getFusedGroupCount() == (specialize ? 1u : 0u),
                "unexpected forward specialization count");
        void* compressed = nullptr;
        size_t compressedBytes = 0;
        p.compress(deviceInput.ptr, bytes, &compressed, &compressedBytes);
        std::vector<unsigned char> payload(compressedBytes);
        check(cudaMemcpy(payload.data(), compressed, compressedBytes, cudaMemcpyDeviceToHost));
        void* reconstructed = nullptr;
        size_t reconstructedBytes = 0;
        p.decompress(compressed, compressedBytes, &reconstructed, &reconstructedBytes);
        auto output = copyReconstruction(reconstructed, reconstructedBytes, count);
        for (size_t i = 0; i < count; ++i)
            require(std::abs(output[i] - input[i]) <= kBound * 1.001f,
                    "error bound exceeded");
        if (!specialize) {
            stagedPayload = std::move(payload);
            stagedOutput = std::move(output);
        } else {
            require(!p.getFusionInfo().installed_inverse_groups.empty(),
                    "inverse specialization was not installed");
            require(payload == stagedPayload, "specialization changed payload bytes");
            require(std::memcmp(output.data(), stagedOutput.data(), bytes) == 0,
                    "specialization changed reconstruction");
            p.decompress(compressed, compressedBytes, &reconstructed, &reconstructedBytes);
            auto repeated = copyReconstruction(reconstructed, reconstructedBytes, count);
            require(std::memcmp(output.data(), repeated.data(), bytes) == 0,
                    "repeated decompression changed reconstruction");
            p.writeToFile(stem + ".fzm");
            std::ofstream expected(stem + ".bin", std::ios::binary);
            expected.write(reinterpret_cast<const char*>(output.data()), bytes);
            require(expected.good(), "failed to save reference output");
        }
    }
    std::cout << (chunk ? "chunk" : "warp") << " (" << count << " elements): Off/Auto payload identity, "
              << "forward/inverse specialization and bounds passed\n";
}

void read(const std::string& directory, bool chunk, size_t count) {
    const std::string stem = directory + (chunk ? "/chunk" : "/warp") + std::to_string(count);
    std::vector<float> expected(count);
    std::ifstream reference(stem + ".bin", std::ios::binary);
    reference.read(reinterpret_cast<char*>(expected.data()), expected.size() * sizeof(float));
    require(reference.good(), "failed to load reference output");
    void* reconstructed = nullptr;
    size_t bytes = 0;
    Pipeline::decompressFromFile(stem + ".fzm", &reconstructed, &bytes);
    auto output = copyReconstruction(reconstructed, bytes, count);
    check(cudaFree(reconstructed));
    require(std::memcmp(output.data(), expected.data(), bytes) == 0,
            "fresh-process archive reload changed reconstruction");
    std::cout << (chunk ? "chunk" : "warp") << ": fresh-process archive reload passed\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: installed_specialization write|read output-directory");
        const std::string mode = argv[1];
        require(mode == "write" || mode == "read", "invalid mode");
        for (bool chunk : {false, true}) {
            if (mode == "write") {
                write(argv[2], chunk, kPartialCount);
                write(argv[2], chunk, kCount);
            }
            else {
                read(argv[2], chunk, kPartialCount);
                read(argv[2], chunk, kCount);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
