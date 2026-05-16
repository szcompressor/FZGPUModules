// Internal utility — NOT part of the public API.
// Do not include from user or example code.
//
// Adapted from PHF reference (hist.hh)
// Original authors: Cody Rivera, Megan Hickman Fulp
// Changes: removed SEQ/Cauchy variants (unused), replaced include guard with #pragma once.

#pragma once

#include <cstddef>
#include <cstdint>

namespace fz::module {

// Compute kernel launch parameters once at initialization time.
// Stores results in grid_dim, block_dim, shmem_use, r_per_block.
// Must be called from a .cu translation unit (references a __global__ kernel pointer).
template <typename E>
void GPU_histogram_generic_optimizer_on_initialization(
    size_t const data_len, uint16_t const hist_len,
    int& grid_dim, int& block_dim, int& shmem_use, int& r_per_block);

// Launch the p2013 privatized histogram kernel.
// Caller is responsible for zeroing out_hist before the call.
template <typename E>
int GPU_histogram_generic(
    E* in_data, size_t const data_len, uint32_t* out_hist, uint16_t const hist_len,
    int const grid_dim, int const block_dim, int const shmem_use, int const r_per_block,
    void* stream);

}  // namespace fz::module
