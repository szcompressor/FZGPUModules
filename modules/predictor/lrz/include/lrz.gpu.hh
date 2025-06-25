/**
 * @file l23.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-01
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef FZ_MODULE_LRZ_GPU_HH
#define FZ_MODULE_LRZ_GPU_HH

#include <array>
#include <cstdint>

// #include "type.h"

#define PROPER_EB double

using stdlen3 = std::array<size_t, 3>;

namespace fz::module {

template <typename T, bool UseZigZag, typename Eq>
int GPU_c_lorenzo_nd_with_outlier(T* const in_data, stdlen3 const _data_len3,
                                  Eq* const out_eq, T* outlier_vals,
                                  uint32_t* outlier_idxs,
                                  uint32_t* num_outliers, uint32_t* out_top1,
                                  double const ebx2, double const ebx2_r,
                                  uint16_t const radius, void* stream);

template <typename T, bool UseZigZag, typename Eq>
int GPU_x_lorenzo_nd(
    Eq* const in_eq, T* const in_outlier, T* const out_data, stdlen3 const _data_len3,
    double const ebx2, double const ebx2_r, uint16_t const radius, void* stream);

template <typename TIN, typename TOUT, bool ReverseProcess>
int GPU_lorenzo_prequant(
    TIN* const in, size_t const len, double const ebx2, double const ebx2_r, TOUT* const out,
    void* _stream);

}   // namespace fz::module

#endif /* FZ_MODULE_LRZ_GPU_HH */