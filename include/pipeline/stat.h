#pragma once

/**
 * @file stat.h
 * @brief Reconstruction quality metrics (MSE, PSNR, max error, NRMSE).
 */

#include <cstddef>

namespace fz {

/** Per-element quality statistics between original and reconstructed arrays. */
struct ReconstructionStats {
    double mse             = 0.0; ///< Mean squared error.
    double psnr            = 0.0; ///< Peak signal-to-noise ratio (dB).
    double max_error       = 0.0; ///< Maximum absolute element-wise error.
    size_t max_error_index = 0;   ///< Index of the element with maximum error.
    double nrmse           = 0.0; ///< Normalized RMSE (RMSE / value_range).
    double value_range     = 0.0; ///< max(data) − min(data) of the original.
};

/**
 * Compute reconstruction statistics between two device arrays.
 *
 * @param d_original      Device pointer to original data.
 * @param d_decompressed  Device pointer to reconstructed data.
 * @param n               Number of elements.
 */
template<typename T>
ReconstructionStats calculateStatistics(const T* d_original, const T* d_decompressed, size_t n);

} // namespace fz
