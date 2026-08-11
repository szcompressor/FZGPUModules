/**
 * modules/structural/roibin_split/roibin_split_stage.cu
 *
 * ROIBinSplitStage — forward: field → (roi boxes, binned background, peak table).
 * Inverse: those three → field. See roibin_split_stage.h for the design rationale,
 * in particular why the ROI stream stores overlapping box pixels redundantly
 * instead of running a device-wide stream compaction, and why binning is a
 * resolution reduction rather than an error bound.
 */

#include "structural/roibin_split/roibin_split_stage.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"

#include "backend/api.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace fz {

namespace {

constexpr int kBlock = 256;

/// Clamp helper — boxes at frame edges are clamped, not truncated, so the box
/// size stays fixed (see header). Duplicate reads/writes are idempotent.
__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/**
 * Gather the (2*hw+1)^2 box around each peak into `roi`, peak-major.
 * One thread per ROI slot; slot i belongs to peak i/box.
 */
template <typename T>
__global__ void k_extract_roi(const T* __restrict__ field,
                              const RoiPeak* __restrict__ peaks,
                              T* __restrict__ roi,
                              uint32_t npeaks, uint32_t side, uint32_t nx,
                              uint32_t ny, uint32_t nz, int hw)
{
    const size_t total = size_t(npeaks) * side * side;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
         i < total; i += size_t(gridDim.x) * blockDim.x) {
        const uint32_t p   = static_cast<uint32_t>(i / (size_t(side) * side));
        const uint32_t rem = static_cast<uint32_t>(i % (size_t(side) * side));
        const int dy = int(rem / side) - hw;
        const int dx = int(rem % side) - hw;

        const RoiPeak pk = peaks[p];
        const int x = clampi(int(pk.x) + dx, 0, int(nx) - 1);
        const int y = clampi(int(pk.y) + dy, 0, int(ny) - 1);
        const uint32_t z = pk.z < nz ? pk.z : nz - 1;

        roi[i] = field[(size_t(z) * ny + y) * nx + x];
    }
}

/// Scatter ROI boxes back over the (already un-binned) field.
template <typename T>
__global__ void k_scatter_roi(T* __restrict__ field,
                              const RoiPeak* __restrict__ peaks,
                              const T* __restrict__ roi,
                              uint32_t npeaks, uint32_t side, uint32_t nx,
                              uint32_t ny, uint32_t nz, int hw)
{
    const size_t total = size_t(npeaks) * side * side;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
         i < total; i += size_t(gridDim.x) * blockDim.x) {
        const uint32_t p   = static_cast<uint32_t>(i / (size_t(side) * side));
        const uint32_t rem = static_cast<uint32_t>(i % (size_t(side) * side));
        const int dy = int(rem / side) - hw;
        const int dx = int(rem % side) - hw;

        const RoiPeak pk = peaks[p];
        const int x = clampi(int(pk.x) + dx, 0, int(nx) - 1);
        const int y = clampi(int(pk.y) + dy, 0, int(ny) - 1);
        const uint32_t z = pk.z < nz ? pk.z : nz - 1;

        field[(size_t(z) * ny + y) * nx + x] = roi[i];
    }
}

/**
 * Box-average each bin x bin block of every z-slice into `bg`.
 * Edge blocks average only the pixels that exist, so the mean is unbiased at the
 * frame border rather than being pulled toward a padding value.
 */
template <typename T>
__global__ void k_bin(const T* __restrict__ field, T* __restrict__ bg,
                      uint32_t nx, uint32_t ny, uint32_t nz,
                      uint32_t bnx, uint32_t bny, uint32_t bin)
{
    const size_t total = size_t(bnx) * bny * nz;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
         i < total; i += size_t(gridDim.x) * blockDim.x) {
        const uint32_t z  = static_cast<uint32_t>(i / (size_t(bnx) * bny));
        const uint32_t r  = static_cast<uint32_t>(i % (size_t(bnx) * bny));
        const uint32_t by = r / bnx;
        const uint32_t bx = r % bnx;

        const uint32_t x0 = bx * bin, y0 = by * bin;
        const uint32_t x1 = min(x0 + bin, nx), y1 = min(y0 + bin, ny);

        double acc = 0.0;
        uint32_t cnt = 0;
        for (uint32_t y = y0; y < y1; ++y)
            for (uint32_t x = x0; x < x1; ++x) {
                acc += static_cast<double>(field[(size_t(z) * ny + y) * nx + x]);
                ++cnt;
            }
        bg[i] = static_cast<T>(cnt ? acc / cnt : 0.0);
    }
}

/// Replicate each background value across its bin x bin block.
template <typename T>
__global__ void k_unbin(const T* __restrict__ bg, T* __restrict__ field,
                        uint32_t nx, uint32_t ny, uint32_t nz,
                        uint32_t bnx, uint32_t bny, uint32_t bin)
{
    const size_t total = size_t(nx) * ny * nz;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
         i < total; i += size_t(gridDim.x) * blockDim.x) {
        const uint32_t z = static_cast<uint32_t>(i / (size_t(nx) * ny));
        const uint32_t r = static_cast<uint32_t>(i % (size_t(nx) * ny));
        const uint32_t y = r / nx;
        const uint32_t x = r % nx;
        field[i] = bg[(size_t(z) * bny + (y / bin)) * bnx + (x / bin)];
    }
}

inline int gridFor(size_t n) {
    size_t g = (n + kBlock - 1) / kBlock;
    if (g == 0) g = 1;
    return static_cast<int>(g > 65535 ? 65535 : g);
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────

template <typename TData>
void ROIBinSplitStage<TData>::setPeaks(const std::vector<RoiPeak>& peaks) {
    peaks_ = peaks;
    computeOverlapFraction();
}

template <typename TData>
void ROIBinSplitStage<TData>::setPeaksFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("ROIBinSplit: cannot open peaks file: " + path);

    char magic[8] = {0};
    f.read(magic, 8);
    if (std::string(magic, 6) != "FZROI1")
        throw std::runtime_error("ROIBinSplit: bad magic in peaks file: " + path);

    uint32_t nx = 0, ny = 0, nz = 0, npeaks = 0;
    f.read(reinterpret_cast<char*>(&nx), 4);
    f.read(reinterpret_cast<char*>(&ny), 4);
    f.read(reinterpret_cast<char*>(&nz), 4);
    f.read(reinterpret_cast<char*>(&npeaks), 4);
    if (!f) throw std::runtime_error("ROIBinSplit: truncated peaks header: " + path);

    // The peak file records the geometry it was produced for. If the pipeline was
    // given dimensions too, they must agree — a silent mismatch would place every
    // ROI box on the wrong pixels and still produce plausible-looking output.
    if (dims_[0] && (nx != dims_[0] || ny != dims_[1] || nz != dims_[2])) {
        throw std::runtime_error(
            "ROIBinSplit: peaks file geometry " + std::to_string(nx) + "x" +
            std::to_string(ny) + "x" + std::to_string(nz) +
            " does not match pipeline dims " + std::to_string(dims_[0]) + "x" +
            std::to_string(dims_[1]) + "x" + std::to_string(dims_[2]));
    }
    if (!dims_[0]) dims_ = {nx, ny, nz};

    peaks_.resize(npeaks);
    if (npeaks) {
        f.read(reinterpret_cast<char*>(peaks_.data()),
               static_cast<std::streamsize>(npeaks * sizeof(RoiPeak)));
        if (!f) throw std::runtime_error("ROIBinSplit: truncated peak records: " + path);
    }

    for (size_t i = 0; i < peaks_.size(); ++i) {
        const RoiPeak& p = peaks_[i];
        if (p.x >= nx || p.y >= ny || p.z >= nz)
            throw std::runtime_error(
                "ROIBinSplit: peak " + std::to_string(i) + " at (" +
                std::to_string(p.x) + "," + std::to_string(p.y) + "," +
                std::to_string(p.z) + ") is outside the field");
    }

    computeOverlapFraction();
    FZ_LOG(DEBUG, "ROIBinSplit: loaded %zu peaks from %s (%ux%ux%u)",
           peaks_.size(), path.c_str(), nx, ny, nz);
}

template <typename TData>
void ROIBinSplitStage<TData>::computeOverlapFraction() {
    const size_t side = 2 * half_width_ + 1;
    const size_t total = peaks_.size() * side * side;
    if (!total || !dims_[0]) { overlap_frac_ = 0.0; return; }

    std::unordered_set<uint64_t> seen;
    seen.reserve(total * 2);
    size_t dup = 0;
    for (const RoiPeak& p : peaks_) {
        for (size_t r = 0; r < side * side; ++r) {
            long dy = long(r / side) - long(half_width_);
            long dx = long(r % side) - long(half_width_);
            long x = long(p.x) + dx, y = long(p.y) + dy;
            x = x < 0 ? 0 : (x >= long(dims_[0]) ? long(dims_[0]) - 1 : x);
            y = y < 0 ? 0 : (y >= long(dims_[1]) ? long(dims_[1]) - 1 : y);
            const uint64_t key = (uint64_t(p.z) << 42) ^ (uint64_t(y) << 21) ^ uint64_t(x);
            if (!seen.insert(key).second) ++dup;
        }
    }
    overlap_frac_ = double(dup) / double(total);
}

template <typename TData>
std::vector<std::string> ROIBinSplitStage<TData>::getRunNotes() const {
    std::vector<std::string> notes;
    char buf[256];
    const size_t nel = dims_[0] * dims_[1] * dims_[2];
    std::snprintf(buf, sizeof(buf),
                  "ROIBinSplit: %zu peaks, box %ux%u, roi %zu elems (%.4f%% of field, "
                  "%.2f%% duplicated), bin %u -> bg %zu elems (%.2f%% of field)",
                  peaks_.size(), 2 * half_width_ + 1, 2 * half_width_ + 1,
                  getRoiCount(), nel ? 100.0 * double(getRoiCount()) / double(nel) : 0.0,
                  100.0 * overlap_frac_, bin_,
                  getBgCount(), nel ? 100.0 * double(getBgCount()) / double(nel) : 0.0);
    notes.emplace_back(buf);
    if (bin_ > 1)
        notes.emplace_back(
            "ROIBinSplit: bin_factor > 1 — background error is binning error plus "
            "quantization error and is NOT bounded by the background branch's error "
            "bound; report background fidelity as PSNR only.");
    return notes;
}

template <typename TData>
void ROIBinSplitStage<TData>::onFinalize(size_t /*estimated_inlen*/, MemoryPool* pool) {
    // Forward only: the inverse path receives the table on the `peaks` input port.
    if (is_inverse_ || peaks_.empty() || !pool) return;
    const size_t bytes = peaks_.size() * sizeof(RoiPeak);
    d_peaks_ = static_cast<RoiPeak*>(pool->allocatePersistentDevice(bytes, "roibin_peaks"));
    FZ_CUDA_CHECK(cudaMemcpy(d_peaks_, peaks_.data(), bytes, cudaMemcpyHostToDevice));
}

template <typename TData>
void ROIBinSplitStage<TData>::execute(
    cudaStream_t stream,
    MemoryPool* /*pool*/,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    const uint32_t nx = static_cast<uint32_t>(dims_[0]);
    const uint32_t ny = static_cast<uint32_t>(dims_[1]);
    const uint32_t nz = static_cast<uint32_t>(dims_[2]);
    if (!nx || !ny || !nz)
        throw std::runtime_error("ROIBinSplit: dimensions not set (call setDims)");

    const uint32_t side   = 2 * half_width_ + 1;
    const uint32_t npeaks = static_cast<uint32_t>(peaks_.size());
    const uint32_t bnx = static_cast<uint32_t>(getBgNx());
    const uint32_t bny = static_cast<uint32_t>(getBgNy());
    const size_t   nel = size_t(nx) * ny * nz;

    if (!is_inverse_) {
        if (inputs.size() < 1 || outputs.size() < 3)
            throw std::runtime_error("ROIBinSplit: forward needs 1 input and 3 outputs");
        if (sizes[0] < nel * sizeof(TData))
            throw std::runtime_error("ROIBinSplit: input smaller than nx*ny*nz");

        const TData* field = static_cast<const TData*>(inputs[0]);
        TData* roi = static_cast<TData*>(outputs[0]);
        TData* bg  = static_cast<TData*>(outputs[1]);

        if (npeaks) {
            if (!d_peaks_)
                throw std::runtime_error("ROIBinSplit: peak table not uploaded "
                                         "(onFinalize did not run)");
            const size_t nroi = size_t(npeaks) * side * side;
            k_extract_roi<TData><<<gridFor(nroi), kBlock, 0, stream>>>(
                field, d_peaks_, roi, npeaks, side, nx, ny, nz, int(half_width_));
            FZ_CUDA_CHECK(cudaGetLastError());

            // Emit the peak table so the archive is self-contained.
            FZ_CUDA_CHECK(cudaMemcpyAsync(outputs[2], d_peaks_,
                                          size_t(npeaks) * sizeof(RoiPeak),
                                          cudaMemcpyDeviceToDevice, stream));
        }

        if (bin_ == 1) {
            FZ_CUDA_CHECK(cudaMemcpyAsync(bg, field, nel * sizeof(TData),
                                          cudaMemcpyDeviceToDevice, stream));
        } else {
            k_bin<TData><<<gridFor(getBgCount()), kBlock, 0, stream>>>(
                field, bg, nx, ny, nz, bnx, bny, bin_);
            FZ_CUDA_CHECK(cudaGetLastError());
        }

    } else {
        if (inputs.size() < 3 || outputs.size() < 1)
            throw std::runtime_error("ROIBinSplit: inverse needs 3 inputs and 1 output");

        const TData* roi = static_cast<const TData*>(inputs[0]);
        const TData* bg  = static_cast<const TData*>(inputs[1]);
        const RoiPeak* peaks = static_cast<const RoiPeak*>(inputs[2]);
        TData* field = static_cast<TData*>(outputs[0]);

        // Background first, then paste the ROI boxes over it — the ROI values are
        // the tight-bound ones and must win wherever the two overlap.
        if (bin_ == 1) {
            FZ_CUDA_CHECK(cudaMemcpyAsync(field, bg, nel * sizeof(TData),
                                          cudaMemcpyDeviceToDevice, stream));
        } else {
            k_unbin<TData><<<gridFor(nel), kBlock, 0, stream>>>(
                bg, field, nx, ny, nz, bnx, bny, bin_);
            FZ_CUDA_CHECK(cudaGetLastError());
        }

        if (npeaks) {
            const size_t nroi = size_t(npeaks) * side * side;
            k_scatter_roi<TData><<<gridFor(nroi), kBlock, 0, stream>>>(
                field, peaks, roi, npeaks, side, nx, ny, nz, int(half_width_));
            FZ_CUDA_CHECK(cudaGetLastError());
        }
    }
}

template class ROIBinSplitStage<float>;
template class ROIBinSplitStage<double>;

} // namespace fz
