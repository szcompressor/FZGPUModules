// Bounded compression-latency probe; interpretation belongs in CN-GRAPH-1.
#include "fzgpumodules.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static void check(cudaError_t status) {
    if(status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size()/2];
}

int main(int argc, char** argv) {
    if(argc != 6 && argc != 7) {
        std::cerr << "Usage: graph_capture_comparison input.f32 elements lorenzo|lorenzo_rle|demo minimal|preallocate|graph|auto repeats [same-input]\n";
        return 2;
    }
    try {
        const size_t count = std::stoull(argv[2]);
        const std::string family = argv[3], mode = argv[4];
        const int repeats = std::stoi(argv[5]);
        const bool change_input = argc == 6;
        if(argc == 7 && std::string(argv[6]) != "same-input") throw std::invalid_argument("Unknown validation option");
        if(!count || repeats < 1 || (family != "lorenzo" && family != "lorenzo_rle" && family != "demo") ||
           (mode != "minimal" && mode != "preallocate" && mode != "graph" && mode != "auto"))
            throw std::invalid_argument("Invalid size, pipeline, mode, or repetition count");
        std::vector<float> input(count);
        std::ifstream file(argv[1], std::ios::binary);
        if(!file.read(reinterpret_cast<char*>(input.data()), count*sizeof(float)))
            throw std::runtime_error("Cannot read requested f32 input");
        for(float x: input) if(!std::isfinite(x)) throw std::runtime_error("Nonfinite input");
        const size_t bytes = count*sizeof(float);
        constexpr double bound = 1.0/1024;
        cudaStream_t stream;
        check(cudaStreamCreate(&stream));
        void* device = nullptr;
        check(cudaMalloc(&device, bytes));
        check(cudaMemcpy(device, input.data(), bytes, cudaMemcpyHostToDevice));
        cudaEvent_t begin, end;
        check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        const auto setup_start = Clock::now();
        auto pipeline = std::make_unique<fz::Pipeline>(bytes,
            mode == "minimal" ? fz::MemoryStrategy::MINIMAL : fz::MemoryStrategy::PREALLOCATE, 8.0f);
        if(family == "demo") {
            auto* q = pipeline->addStage<fz::QuantizerStage<float, uint32_t>>();
            q->setLinearMode(true); q->setErrorBound(bound);
            auto* l = pipeline->addStage<fz::LorenzoStage<int32_t>>(); l->setBlockSize(32);
            auto* b = pipeline->addStage<fz::AdaptiveBitpackStage<int32_t>>(); b->setBlockSize(32);
            pipeline->connect(l, q, "codes"); pipeline->connect(b, l);
        } else {
            auto* l = pipeline->addStage<fz::LorenzoQuantStage<float, uint16_t>>();
            l->setErrorBound(bound); l->setQuantRadius(512); l->setOutlierCapacity(1.0f);
            if(family == "lorenzo_rle") {
                auto* r = pipeline->addStage<fz::RLEStage<uint16_t>>();
                pipeline->connect(r, l, "codes");
            }
        }
        pipeline->setSpecializationPolicy(mode == "auto" ? fz::SpecializationPolicy::Auto : fz::SpecializationPolicy::Off);
        if(mode == "graph") pipeline->enableGraphMode(true);
        pipeline->finalize();
        if(mode == "graph") { pipeline->warmup(stream); pipeline->captureGraph(stream); }
        check(cudaStreamSynchronize(stream));
        const double setup_ms = ms(setup_start);
        for(int i=0; i<3; ++i) { pipeline->compress({device, bytes}, stream); check(cudaStreamSynchronize(stream)); }
        std::vector<double> host_ms, gpu_ms;
        for(int i=0; i<repeats; ++i) {
            const auto start = Clock::now();
            check(cudaEventRecord(begin, stream));
            pipeline->compress({device, bytes}, stream);
            check(cudaEventRecord(end, stream));
            check(cudaEventSynchronize(end));
            host_ms.push_back(ms(start));
            float elapsed; check(cudaEventElapsedTime(&elapsed, begin, end)); gpu_ms.push_back(elapsed);
        }
        double max_error = 0;
        size_t compressed_bytes = 0;
        uint64_t reconstructed_hash = 14695981039346656037ULL;
        // Also change the input after replay and decode, to detect stale captured data/metadata.
        for(int variant=0; variant<2; ++variant) {
            if(variant && change_input) std::rotate(input.begin(), input.begin()+std::min<size_t>(7, count), input.end());
            check(cudaMemcpy(device, input.data(), bytes, cudaMemcpyHostToDevice));
            std::cerr << "Validating input " << variant << "\n";
            auto compressed = pipeline->compress({device, bytes}, stream);
            compressed_bytes = compressed.bytes();
            auto restored = pipeline->decompressOwned({compressed.data(), compressed.bytes()}, stream);
            if(restored.bytes() != bytes) throw std::runtime_error("Decoded size changed on validation input " + std::to_string(variant));
            std::vector<float> output(count);
            check(cudaMemcpyAsync(output.data(), restored.data(), bytes, cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream));
            for(size_t i=0; i<count; ++i) {
                if(!std::isfinite(output[i])) throw std::runtime_error("Nonfinite reconstruction");
                max_error = std::max(max_error, std::abs(double(output[i])-input[i]));
            }
            if(max_error > bound) throw std::runtime_error("Reconstruction exceeds absolute bound on validation input " + std::to_string(variant));
            if(!variant) {
                const auto* raw = reinterpret_cast<const uint8_t*>(output.data());
                for(size_t i=0; i<bytes; ++i) {
                    reconstructed_hash ^= raw[i]; reconstructed_hash *= 1099511628211ULL;
                }
            }
        }
        std::cout << "RESULT {\"elements\":" << count << ",\"pipeline\":\"" << family << "\",\"mode\":\"" << mode
                  << "\",\"host_ms\":" << median(host_ms) << ",\"gpu_ms\":" << median(gpu_ms)
                  << ",\"setup_ms\":" << setup_ms << ",\"peak_pool_bytes\":" << pipeline->getPeakMemoryUsage()
                  << ",\"archive_bytes\":" << compressed_bytes << ",\"max_abs_error\":" << max_error
                  << ",\"specialized_groups\":" << pipeline->getSpecializedGroupCount()
                  << ",\"reconstruction_hash\":\"" << reconstructed_hash << "\",\"changed_input_verified\":" << (change_input ? "true" : "false") << "}\n";
        pipeline.reset();
        check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
        check(cudaFree(device)); check(cudaStreamDestroy(stream));
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
