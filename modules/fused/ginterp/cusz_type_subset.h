#pragma once

/**
 * @file cusz_type_subset.h
 * @brief Minimal subset of cuSZ-Hi's `cusz/type.h` needed by the ported
 *        spline interpolation kernels in `ginterp_md.inl`.
 *
 * `ginterp_md.inl` references `u4` and `INTERPOLATION_PARAMS` both inside
 * and outside its primary namespace, so both symbols are defined at file
 * scope here (matching the original cuSZ-Hi layout). They are only meant to
 * be visible inside the GInterp translation unit — do NOT include this
 * header from public stage headers.
 *
 * Source: cuSZ-Hi (https://github.com/shixun404/cuSZ-Hi), BSD-3-Clause.
 * See the cuSZ-Hi repository and `THIRD_PARTY.md` for the upstream source.
 */

#include <cstdint>

using u4 = uint32_t;

/**
 * Interpolation parameter bundle consumed by the spline encode / decode
 * kernels. Layout matches the cuSZ-Hi upstream definition so the kernels
 * compile unchanged. Default values reproduce the cuSZ-Hi deterministic
 * baseline (`auto_tuning = 0`): cubic spline with `alpha=1.75`, `beta=4.0`,
 * no multi-dimensional or natural variants engaged.
 *
 * `use_md`, `use_natural`, and `reverse` are indexed by interpolation level
 * (0 = finest, LEVEL-1 = coarsest). The MVP uses LEVEL=4 for the 3D path.
 */
struct INTERPOLATION_PARAMS {
    double alpha{1.75};
    double beta{4.0};
    bool use_md[6];
    bool use_natural[6];
    bool reverse[6];
    uint8_t auto_tuning{0};

    INTERPOLATION_PARAMS()
        : use_md{true, true, false, false, false, false},
          use_natural{false, false, false, false, false, false},
          reverse{false, false, false, false, false, false} {}
};
