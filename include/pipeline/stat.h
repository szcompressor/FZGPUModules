#pragma once

#include <cstddef>

namespace fz {

/**
 * Statistics for reconstruction accuracy
 */
struct ReconstructionStats {
    double mse = 0.0;
    double psnr = 0.0;
    double max_error = 0.0;
    size_t max_error_index = 0;
    double nrmse = 0.0;
    double value_range = 0.0;
};

/**
 * Calculate reconstruction statistics (MSE, PSNR, Max Error) between original and decompressed data.
 * @param d_original Device pointer to original uncompressed data
 * @param d_decompressed Device pointer to decompressed data
 * @param n Number of elements
 * @return Reconstruction statistics
 */
template<typename T>
ReconstructionStats calculateStatistics(const T* d_original, const T* d_decompressed, size_t n);

} // namespace fz
