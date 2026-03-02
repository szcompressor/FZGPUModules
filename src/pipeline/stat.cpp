#include "pipeline/stat.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace fz {

template<typename T>
ReconstructionStats calculateStatistics(const T* d_original, const T* d_decompressed, size_t n) {
    if (n == 0) return ReconstructionStats();
    
    std::vector<T> h_orig(n);
    std::vector<T> h_decomp(n);
    cudaMemcpy(h_orig.data(), d_original, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_decomp.data(), d_decompressed, n * sizeof(T), cudaMemcpyDeviceToHost);
    
    double sum_sq_err = 0;
    double max_err = 0;
    size_t max_err_idx = 0;
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    
    for(size_t i = 0; i < n; i++) {
        double orig = static_cast<double>(h_orig[i]);
        double decomp = static_cast<double>(h_decomp[i]);
        double err = std::abs(orig - decomp);
        
        sum_sq_err += err * err;
        
        if (err > max_err) {
            max_err = err;
            max_err_idx = i;
        }
        
        if (orig < min_val) min_val = orig;
        if (orig > max_val) max_val = orig;
    }
    
    ReconstructionStats stats;
    stats.mse = sum_sq_err / n;
    stats.max_error = max_err;
    stats.max_error_index = max_err_idx;
    stats.value_range = max_val - min_val;
    
    if (stats.value_range > 0 && stats.mse > 0) {
        stats.psnr = 20 * std::log10(stats.value_range) - 10 * std::log10(stats.mse);
        stats.nrmse = std::sqrt(stats.mse) / stats.value_range;
    } else if (stats.mse == 0) {
        stats.psnr = std::numeric_limits<double>::infinity();
        stats.nrmse = 0.0;
    } else {
        stats.psnr = 0.0;
        stats.nrmse = 0.0;
    }
    
    return stats;
}

// Explicit instantiations for common types
template ReconstructionStats calculateStatistics<float>(const float*, const float*, size_t);
template ReconstructionStats calculateStatistics<double>(const double*, const double*, size_t);
template ReconstructionStats calculateStatistics<int>(const int*, const int*, size_t);
template ReconstructionStats calculateStatistics<long long>(const long long*, const long long*, size_t);

} // namespace fz
