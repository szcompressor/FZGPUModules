#include "fzgpumodules.h"
#include "pipeline/stat.h"
#include <cmath>
#include <iostream>

using namespace fz;
static constexpr size_t N = 1024 * 1024;

static float* make_data() {
    constexpr size_t NX=1024, NY=1024;
    std::vector<float> h(N);
    for (size_t i=0;i<NY;i++) for (size_t j=0;j<NX;j++)
        h[i*NX+j] = std::cos(2.f*3.14159f*3*(float)j/NX)*std::sin(2.f*3.14159f*2*(float)i/NY)*80+200;
    float* d; cudaMalloc(&d, N*sizeof(float));
    cudaMemcpy(d,h.data(),N*sizeof(float),cudaMemcpyHostToDevice);
    return d;
}

int main(){
    float* d_in = make_data();
    size_t nb = N*sizeof(float);

    // Test 1: Quantizer + ABS mode round trip
    {
        Pipeline p(nb,MemoryStrategy::PREALLOCATE,3.f);
        auto* q = p.addStage<QuantizerStage<float,uint32_t>>();
        q->setErrorBound(1e-3f);
        q->setErrorBoundMode(ErrorBoundMode::REL);
        q->setQuantRadius(32768);
        q->setOutlierCapacity(0.05f);
        p.finalize();
        void* comp; size_t csz;
        p.compress(d_in, nb, &comp, &csz, 0);
        cudaDeviceSynchronize();
        void* decomp; size_t dsz;
        p.decompress(comp, csz, &decomp, &dsz, 0);
        cudaDeviceSynchronize();
        if (dsz == nb) {
            auto s = calculateStatistics<float>(d_in,(const float*)decomp,N);
            std::cout << "Quant only PSNR=" << s.psnr << " maxerr=" << s.max_error << "\n";
        } else {
            std::cout << "Quant only: size mismatch " << dsz << " vs " << nb << "\n";
        }
        cudaFree(decomp);
    }

    // Test 2: Quantizer + Diff<int32,uint32> round trip
    {
        Pipeline p(nb,MemoryStrategy::PREALLOCATE,3.f);
        auto* q = p.addStage<QuantizerStage<float,uint32_t>>();
        q->setErrorBound(1e-3f);
        q->setErrorBoundMode(ErrorBoundMode::REL);
        q->setQuantRadius(32768);
        q->setOutlierCapacity(0.05f);
        auto* d = p.addStage<DifferenceStage<int32_t,uint32_t>>();
        d->setChunkSize(16384);
        p.connect(d,q,"codes");
        p.finalize();
        void* comp; size_t csz;
        p.compress(d_in, nb, &comp, &csz, 0);
        cudaDeviceSynchronize();
        void* decomp; size_t dsz;
        p.decompress(comp, csz, &decomp, &dsz, 0);
        cudaDeviceSynchronize();
        if (dsz == nb) {
            auto s = calculateStatistics<float>(d_in,(const float*)decomp,N);
            std::cout << "Quant+Diff<i32,u32> PSNR=" << s.psnr << " maxerr=" << s.max_error << "\n";
        } else {
            std::cout << "Quant+Diff<i32,u32>: size mismatch\n";
        }
        cudaFree(decomp);
    }

    // Test 3: Quantizer + Diff<uint32,uint32> round trip
    {
        Pipeline p(nb,MemoryStrategy::PREALLOCATE,3.f);
        auto* q = p.addStage<QuantizerStage<float,uint32_t>>();
        q->setErrorBound(1e-3f);
        q->setErrorBoundMode(ErrorBoundMode::REL);
        q->setQuantRadius(32768);
        q->setOutlierCapacity(0.05f);
        auto* d = p.addStage<DifferenceStage<uint32_t>>();
        d->setChunkSize(16384);
        p.connect(d,q,"codes");
        p.finalize();
        void* comp; size_t csz;
        p.compress(d_in, nb, &comp, &csz, 0);
        cudaDeviceSynchronize();
        void* decomp; size_t dsz;
        p.decompress(comp, csz, &decomp, &dsz, 0);
        cudaDeviceSynchronize();
        if (dsz == nb) {
            auto s = calculateStatistics<float>(d_in,(const float*)decomp,N);
            std::cout << "Quant+Diff<u32> PSNR=" << s.psnr << " maxerr=" << s.max_error << "\n";
        } else {
            std::cout << "Quant+Diff<u32>: size mismatch\n";
        }
        cudaFree(decomp);
    }
    cudaFree(d_in);
    return 0;
}

// Test ABS: should give good quality
// (appended for diagnosis)
