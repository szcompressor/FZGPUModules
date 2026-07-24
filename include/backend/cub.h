#pragma once

/**
 * @file include/backend/cub.h
 * @brief Backend-neutral access to cub's block- and device-level primitives.
 *
 * hipCUB is a close mirror of cub: same algorithm names, same two-call
 * (size-then-run) shape, same block-level collectives — it differs only in
 * header root (<hipcub/hipcub.hpp> vs <cub/cub.cuh>) and namespace (`hipcub`
 * vs `cub`). Aliasing the namespace here lets the existing `cub::DeviceScan`,
 * `cub::BlockScan`, `cub::DeviceReduce`, `cub::DeviceRadixSort`, and
 * `cub::BlockRadixSort` call sites compile unchanged on both backends.
 *
 * Include this instead of <cub/...> directly. Doing so matters beyond
 * tidiness: on systems where a CUDA toolkit is also installed and on the
 * compiler's default include path (CPATH — the usual setup on Cray/HPE
 * machines with a `cudatoolkit` module loaded), a bare `#include <cub/cub.cuh>`
 * silently resolves to NVIDIA's cub during a HIP build and drags the whole
 * CUDA runtime in with it.
 *
 * The scratch-storage boilerplate that wraps the device-level calls lives in
 * backend/algorithms.h; this header is just the include + namespace.
 */

#if defined(FZGMOD_BACKEND_HIP)

#include <hipcub/hipcub.hpp>

/// Lets `cub::`-qualified call sites resolve to hipCUB under the HIP backend.
namespace cub = hipcub;

#elif defined(FZGMOD_BACKEND_SYCL)

#error "FZGMOD_BACKEND=SYCL is not implemented yet"

#else // FZGMOD_BACKEND_CUDA (default)

#include <cub/cub.cuh>

#endif
