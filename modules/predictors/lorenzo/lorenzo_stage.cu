#include "predictors/lorenzo/lorenzo_stage.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace fz {

// ─────────────────────────────────────────────────────────────────────────────
// 1-D kernels
// ─────────────────────────────────────────────────────────────────────────────

// Forward: d_output[i] = d_input[i] - d_input[i-1]  (d_input[-1] == 0)
// Blocks are independent — each block restarts its chain at 0.
// This matches the block-local model used by LorenzoQuantStage so that the
// inverse (prefix sum) is fully self-contained per block.
template<typename T>
__global__ void lorenzo_delta_1d_kernel(
    const T* __restrict__ in,
    T*       __restrict__ out,
    size_t n)
{
    const size_t block_offset = static_cast<size_t>(blockIdx.x) * blockDim.x;
    const size_t gid = block_offset + threadIdx.x;
    if (gid >= n) return;

    // Previous element: 0 at the start of each block.
    T prev = (threadIdx.x > 0) ? in[gid - 1] : static_cast<T>(0);
    out[gid] = in[gid] - prev;
}

// Inverse: parallel prefix sum (exclusive scan within each block)
// Same block-local model: each block's input is already self-contained.
template<typename T>
__global__ void lorenzo_scan_1d_kernel(
    const T* __restrict__ in,
    T*       __restrict__ out,
    size_t n)
{
    extern __shared__ char smem[];
    T* s = reinterpret_cast<T*>(smem);

    const size_t block_offset = static_cast<size_t>(blockIdx.x) * blockDim.x;
    const size_t gid = block_offset + threadIdx.x;
    const int   tid = static_cast<int>(threadIdx.x);

    s[tid] = (gid < n) ? in[gid] : static_cast<T>(0);
    __syncthreads();

    // Inclusive scan (Hillis-Steele)
    for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1) {
        T val = (tid >= stride) ? s[tid - stride] : static_cast<T>(0);
        __syncthreads();
        s[tid] += val;
        __syncthreads();
    }

    if (gid < n) out[gid] = s[tid];
}

template<typename T>
void launchLorenzoDeltaKernel1D(
    const T* d_input, T* d_output, size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    lorenzo_delta_1d_kernel<T><<<grid, kBlock, 0, stream>>>(d_input, d_output, n);
    FZ_CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launchLorenzoPrefixSumKernel1D(
    const T* d_input, T* d_output, size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    lorenzo_scan_1d_kernel<T>
        <<<grid, kBlock, kBlock * sizeof(T), stream>>>(d_input, d_output, n);
    FZ_CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// 2-D kernels
// ─────────────────────────────────────────────────────────────────────────────

// Forward 2-D: d[x,y] - d[x-1,y] - d[x,y-1] + d[x-1,y-1]
template<typename T>
__global__ void lorenzo_delta_2d_kernel(
    const T* __restrict__ in,
    T*       __restrict__ out,
    size_t nx, size_t ny)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;  // fast dim
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t idx = y * nx + x;
    T v   = in[idx];
    T vx  = (x > 0) ? in[idx - 1]      : static_cast<T>(0);
    T vy  = (y > 0) ? in[idx - nx]     : static_cast<T>(0);
    T vxy = (x > 0 && y > 0) ? in[idx - nx - 1] : static_cast<T>(0);
    out[idx] = v - vx - vy + vxy;
}

// Inverse 2-D: row-first inclusive scan per row, then column-first inclusive
// scan per column.  Two sequential kernel passes suffice and avoid the complex
// 2-D blocked-scan dependency.
template<typename T>
__global__ void lorenzo_scan_row_kernel(
    const T* __restrict__ in,
    T*       __restrict__ out,
    size_t nx, size_t ny)
{
    const size_t y = blockIdx.x;  // one block per row
    if (y >= ny) return;

    extern __shared__ char smem[];
    T* s = reinterpret_cast<T*>(smem);

    const int tid = static_cast<int>(threadIdx.x);
    if (static_cast<size_t>(tid) < nx)
        s[tid] = in[y * nx + tid];
    else
        s[tid] = static_cast<T>(0);
    __syncthreads();

    for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1) {
        T val = (tid >= stride) ? s[tid - stride] : static_cast<T>(0);
        __syncthreads();
        s[tid] += val;
        __syncthreads();
    }

    if (static_cast<size_t>(tid) < nx) out[y * nx + tid] = s[tid];
}

template<typename T>
__global__ void lorenzo_scan_col_kernel(
    const T* __restrict__ in,
    T*       __restrict__ out,
    size_t nx, size_t ny)
{
    const size_t x = blockIdx.x;  // one block per column
    if (x >= nx) return;

    extern __shared__ char smem[];
    T* s = reinterpret_cast<T*>(smem);

    const int tid = static_cast<int>(threadIdx.x);
    if (static_cast<size_t>(tid) < ny)
        s[tid] = in[tid * nx + x];
    else
        s[tid] = static_cast<T>(0);
    __syncthreads();

    for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1) {
        T val = (tid >= stride) ? s[tid - stride] : static_cast<T>(0);
        __syncthreads();
        s[tid] += val;
        __syncthreads();
    }

    if (static_cast<size_t>(tid) < ny) out[tid * nx + x] = s[tid];
}

template<typename T>
void launchLorenzoDeltaKernel2D(
    const T* d_input, T* d_output, size_t nx, size_t ny, cudaStream_t stream)
{
    if (nx == 0 || ny == 0) return;
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((nx + 15) / 16),
              static_cast<unsigned>((ny + 15) / 16));
    lorenzo_delta_2d_kernel<T><<<grid, block, 0, stream>>>(d_input, d_output, nx, ny);
    FZ_CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launchLorenzoPrefixSumKernel2D(
    const T* d_input, T* d_output, size_t nx, size_t ny, cudaStream_t stream)
{
    if (nx == 0 || ny == 0) return;
    // nx and ny must each be <= 1024 for single-pass shared-memory scan.
    // For larger dims the block covers the full extent; shared mem may be large.
    const int bx = static_cast<int>(nx);
    const int by = static_cast<int>(ny);

    // Row scan
    lorenzo_scan_row_kernel<T>
        <<<static_cast<unsigned>(ny), bx, bx * sizeof(T), stream>>>(
            d_input, d_output, nx, ny);
    FZ_CUDA_CHECK(cudaGetLastError());

    // Column scan on row-scan output
    lorenzo_scan_col_kernel<T>
        <<<static_cast<unsigned>(nx), by, by * sizeof(T), stream>>>(
            d_output, d_output, nx, ny);
    FZ_CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-D kernels
// ─────────────────────────────────────────────────────────────────────────────

// Forward 3-D inclusion-exclusion delta (8-neighbor formula)
template<typename T>
__global__ void lorenzo_delta_3d_kernel(
    const T* __restrict__ in,
    T*       __restrict__ out,
    size_t nx, size_t ny, size_t nz)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = z * ny * nx + y * nx + x;
    auto get = [&](ptrdiff_t dx, ptrdiff_t dy, ptrdiff_t dz) -> T {
        ptrdiff_t xx = static_cast<ptrdiff_t>(x) + dx;
        ptrdiff_t yy = static_cast<ptrdiff_t>(y) + dy;
        ptrdiff_t zz = static_cast<ptrdiff_t>(z) + dz;
        if (xx < 0 || yy < 0 || zz < 0) return static_cast<T>(0);
        return in[zz * static_cast<ptrdiff_t>(ny * nx)
                + yy * static_cast<ptrdiff_t>(nx) + xx];
    };

    out[idx] =  get(0,0,0) - get(-1,0,0) - get(0,-1,0) - get(0,0,-1)
              + get(-1,-1,0) + get(-1,0,-1) + get(0,-1,-1)
              - get(-1,-1,-1);
}

// Inverse 3-D: three sequential 1-D prefix sum passes (x → y → z).
template<typename T>
__global__ void lorenzo_scan_x_kernel(
    const T* __restrict__ in, T* __restrict__ out,
    size_t nx, size_t ny, size_t nz)
{
    // one block per (y, z) row
    const size_t y = blockIdx.x % ny;
    const size_t z = blockIdx.x / ny;
    if (y >= ny || z >= nz) return;

    extern __shared__ char smem[];
    T* s = reinterpret_cast<T*>(smem);

    const int tid = static_cast<int>(threadIdx.x);
    const size_t base = z * ny * nx + y * nx;
    if (static_cast<size_t>(tid) < nx) s[tid] = in[base + tid];
    else                               s[tid] = static_cast<T>(0);
    __syncthreads();

    for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1) {
        T val = (tid >= stride) ? s[tid - stride] : static_cast<T>(0);
        __syncthreads();
        s[tid] += val;
        __syncthreads();
    }
    if (static_cast<size_t>(tid) < nx) out[base + tid] = s[tid];
}

template<typename T>
__global__ void lorenzo_scan_y_kernel(
    const T* __restrict__ in, T* __restrict__ out,
    size_t nx, size_t ny, size_t nz)
{
    const size_t x = blockIdx.x % nx;
    const size_t z = blockIdx.x / nx;
    if (x >= nx || z >= nz) return;

    extern __shared__ char smem[];
    T* s = reinterpret_cast<T*>(smem);

    const int tid = static_cast<int>(threadIdx.x);
    if (static_cast<size_t>(tid) < ny)
        s[tid] = in[z * ny * nx + tid * nx + x];
    else
        s[tid] = static_cast<T>(0);
    __syncthreads();

    for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1) {
        T val = (tid >= stride) ? s[tid - stride] : static_cast<T>(0);
        __syncthreads();
        s[tid] += val;
        __syncthreads();
    }
    if (static_cast<size_t>(tid) < ny)
        out[z * ny * nx + tid * nx + x] = s[tid];
}

template<typename T>
__global__ void lorenzo_scan_z_kernel(
    const T* __restrict__ in, T* __restrict__ out,
    size_t nx, size_t ny, size_t nz)
{
    const size_t x = blockIdx.x % nx;
    const size_t y = blockIdx.x / nx;
    if (x >= nx || y >= ny) return;

    extern __shared__ char smem[];
    T* s = reinterpret_cast<T*>(smem);

    const int tid = static_cast<int>(threadIdx.x);
    if (static_cast<size_t>(tid) < nz)
        s[tid] = in[tid * ny * nx + y * nx + x];
    else
        s[tid] = static_cast<T>(0);
    __syncthreads();

    for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1) {
        T val = (tid >= stride) ? s[tid - stride] : static_cast<T>(0);
        __syncthreads();
        s[tid] += val;
        __syncthreads();
    }
    if (static_cast<size_t>(tid) < nz)
        out[tid * ny * nx + y * nx + x] = s[tid];
}

template<typename T>
void launchLorenzoDeltaKernel3D(
    const T* d_input, T* d_output, size_t nx, size_t ny, size_t nz,
    cudaStream_t stream)
{
    if (nx == 0 || ny == 0 || nz == 0) return;
    dim3 block(8, 8, 8);
    dim3 grid(static_cast<unsigned>((nx + 7) / 8),
              static_cast<unsigned>((ny + 7) / 8),
              static_cast<unsigned>((nz + 7) / 8));
    lorenzo_delta_3d_kernel<T><<<grid, block, 0, stream>>>(d_input, d_output, nx, ny, nz);
    FZ_CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launchLorenzoPrefixSumKernel3D(
    const T* d_input, T* d_output, size_t nx, size_t ny, size_t nz,
    cudaStream_t stream)
{
    if (nx == 0 || ny == 0 || nz == 0) return;
    // X-pass: ny*nz blocks, each of nx threads
    lorenzo_scan_x_kernel<T>
        <<<static_cast<unsigned>(ny * nz), static_cast<unsigned>(nx),
           nx * sizeof(T), stream>>>(d_input, d_output, nx, ny, nz);
    FZ_CUDA_CHECK(cudaGetLastError());
    // Y-pass on output of X
    lorenzo_scan_y_kernel<T>
        <<<static_cast<unsigned>(nx * nz), static_cast<unsigned>(ny),
           ny * sizeof(T), stream>>>(d_output, d_output, nx, ny, nz);
    FZ_CUDA_CHECK(cudaGetLastError());
    // Z-pass on output of Y
    lorenzo_scan_z_kernel<T>
        <<<static_cast<unsigned>(nx * ny), static_cast<unsigned>(nz),
           nz * sizeof(T), stream>>>(d_output, d_output, nx, ny, nz);
    FZ_CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// LorenzoStage::execute
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
void LorenzoStage<T>::execute(
    cudaStream_t stream,
    MemoryPool* /*pool*/,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("LorenzoStage: inputs, outputs, and sizes must be non-empty");

    const size_t byte_size = sizes[0];
    if (byte_size == 0) {
        actual_output_size_ = 0;
        return;
    }

    const size_t n = byte_size / sizeof(T);
    const T* in    = static_cast<const T*>(inputs[0]);
    T*       out   = static_cast<T*>(outputs[0]);

    // Resolve effective dims: if dim_x is 0, treat as flat 1-D of n elements.
    size_t nx = (dims_[0] > 0) ? dims_[0] : n;
    size_t ny = dims_[1];
    size_t nz = dims_[2];

    int eff_ndim = ndim();

    if (!is_inverse_) {
        if      (eff_ndim == 3) launchLorenzoDeltaKernel3D<T>(in, out, nx, ny, nz, stream);
        else if (eff_ndim == 2) launchLorenzoDeltaKernel2D<T>(in, out, nx, ny, stream);
        else                    launchLorenzoDeltaKernel1D<T>(in, out, n, stream);
    } else {
        if      (eff_ndim == 3) launchLorenzoPrefixSumKernel3D<T>(in, out, nx, ny, nz, stream);
        else if (eff_ndim == 2) launchLorenzoPrefixSumKernel2D<T>(in, out, nx, ny, stream);
        else                    launchLorenzoPrefixSumKernel1D<T>(in, out, n, stream);
    }

    actual_output_size_ = byte_size;
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit instantiations
// ─────────────────────────────────────────────────────────────────────────────

template class LorenzoStage<int8_t>;
template class LorenzoStage<int16_t>;
template class LorenzoStage<int32_t>;
template class LorenzoStage<int64_t>;

template void launchLorenzoDeltaKernel1D<int8_t> (const int8_t*,  int8_t*,  size_t, cudaStream_t);
template void launchLorenzoDeltaKernel1D<int16_t>(const int16_t*, int16_t*, size_t, cudaStream_t);
template void launchLorenzoDeltaKernel1D<int32_t>(const int32_t*, int32_t*, size_t, cudaStream_t);
template void launchLorenzoDeltaKernel1D<int64_t>(const int64_t*, int64_t*, size_t, cudaStream_t);

template void launchLorenzoPrefixSumKernel1D<int8_t> (const int8_t*,  int8_t*,  size_t, cudaStream_t);
template void launchLorenzoPrefixSumKernel1D<int16_t>(const int16_t*, int16_t*, size_t, cudaStream_t);
template void launchLorenzoPrefixSumKernel1D<int32_t>(const int32_t*, int32_t*, size_t, cudaStream_t);
template void launchLorenzoPrefixSumKernel1D<int64_t>(const int64_t*, int64_t*, size_t, cudaStream_t);

template void launchLorenzoDeltaKernel2D<int8_t> (const int8_t*,  int8_t*,  size_t, size_t, cudaStream_t);
template void launchLorenzoDeltaKernel2D<int16_t>(const int16_t*, int16_t*, size_t, size_t, cudaStream_t);
template void launchLorenzoDeltaKernel2D<int32_t>(const int32_t*, int32_t*, size_t, size_t, cudaStream_t);
template void launchLorenzoDeltaKernel2D<int64_t>(const int64_t*, int64_t*, size_t, size_t, cudaStream_t);

template void launchLorenzoPrefixSumKernel2D<int8_t> (const int8_t*,  int8_t*,  size_t, size_t, cudaStream_t);
template void launchLorenzoPrefixSumKernel2D<int16_t>(const int16_t*, int16_t*, size_t, size_t, cudaStream_t);
template void launchLorenzoPrefixSumKernel2D<int32_t>(const int32_t*, int32_t*, size_t, size_t, cudaStream_t);
template void launchLorenzoPrefixSumKernel2D<int64_t>(const int64_t*, int64_t*, size_t, size_t, cudaStream_t);

template void launchLorenzoDeltaKernel3D<int8_t> (const int8_t*,  int8_t*,  size_t, size_t, size_t, cudaStream_t);
template void launchLorenzoDeltaKernel3D<int16_t>(const int16_t*, int16_t*, size_t, size_t, size_t, cudaStream_t);
template void launchLorenzoDeltaKernel3D<int32_t>(const int32_t*, int32_t*, size_t, size_t, size_t, cudaStream_t);
template void launchLorenzoDeltaKernel3D<int64_t>(const int64_t*, int64_t*, size_t, size_t, size_t, cudaStream_t);

template void launchLorenzoPrefixSumKernel3D<int8_t> (const int8_t*,  int8_t*,  size_t, size_t, size_t, cudaStream_t);
template void launchLorenzoPrefixSumKernel3D<int16_t>(const int16_t*, int16_t*, size_t, size_t, size_t, cudaStream_t);
template void launchLorenzoPrefixSumKernel3D<int32_t>(const int32_t*, int32_t*, size_t, size_t, size_t, cudaStream_t);
template void launchLorenzoPrefixSumKernel3D<int64_t>(const int64_t*, int64_t*, size_t, size_t, size_t, cudaStream_t);

} // namespace fz
