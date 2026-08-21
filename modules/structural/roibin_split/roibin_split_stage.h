#pragma once

/**
 * @file roibin_split_stage.h
 * @brief ROIBinSplitStage — split a detector field into a full-resolution
 *        region-of-interest stream and a (optionally binned) background stream.
 *
 * ## Why this stage exists
 *
 * Serial-crystallography detector frames are almost all background: the science
 * lives in a few hundred Bragg peaks covering well under 1 % of the pixels. A
 * single-bound compressor has to protect those peaks by applying the tight bound
 * to the whole frame, and pays for the other 99 % at that same bound. ROIBIN-SZ
 * (Underwood et al.) attacks this by splitting the frame into peak regions, kept
 * at full resolution, and a background that is spatially binned; the two parts
 * are then compressed separately.
 *
 * @note **Prior work:** this is an independent GPU/DAG implementation of the
 *       ROIBIN-SZ algorithm described by Underwood et al., Synchrotron Radiation
 *       News 36(4), 2023, DOI 10.1080/08940886.2023.2245722. Public reference
 *       integration is in the SZ2 `example/roibin_example` directory; no SZ2 or
 *       LibPressio source was copied. See `THIRD_PARTY.md`.
 *
 * FZGM's DAG can express that split directly: this stage is `1 → 3` forward and
 * `3 → 1` inverse, so the two data streams become two independent branches that
 * can each carry their **own error bound** and their own coder chain, converging
 * again only at the archive (or at a MergeStage). No monolithic GPU compressor
 * applies two error bounds in one pass — expressing it is the point.
 *
 * ```
 *              ┌── roi ──> Quantizer(eb_tight) ──> ... ──> coder ──┐
 *   input ──> split ── bg ──> Quantizer(eb_loose) ──> ... ──> coder ─┤──> archive
 *              └── peaks ────────────────────────────────────────────┘
 * ```
 *
 * ## Where the ROI comes from
 *
 * The peak list is *not* derived from the data. It is the output of the
 * experiment's own peak finder, which in a real light-source pipeline has
 * already run upstream (this is also how ROIBIN-SZ obtains it). At compress time
 * it is read from a `.roi` file via `setPeaksFile()`. It is then emitted on the
 * `peaks` port so it is stored **inside the archive** and counted in the
 * compressed size — the decompressor needs it and must not have to be handed it
 * out of band. At 8 bytes/peak this is ~0.01 % of a frame.
 *
 * ## Geometry and why there is no stream compaction
 *
 * Each peak owns a fixed `(2*hw+1)^2` box, and `roi` is simply those boxes
 * concatenated in peak order. Boxes belonging to nearby peaks may overlap, and
 * overlapping pixels are therefore stored more than once. That redundancy is
 * deliberate: it makes the output size **exactly** `npeaks * box * sizeof(T)`,
 * known before the first kernel launch, so `estimateOutputSizes()` is exact and
 * PREALLOCATE needs no slack. The alternative — a mask plus a device-wide
 * exclusive scan — would need a 4-byte offset per pixel (1.2 GB for the 130-frame
 * volume) to save a redundancy that measures well under 1 % of the ROI stream.
 *
 * Duplicate pixels are safe on the inverse path because scatter is *idempotent*:
 * every copy of a source pixel takes the same value, goes through the same
 * quantizer, and therefore reconstructs to the same number, so the order in
 * which the copies are written back does not matter.
 *
 * Boxes are clamped at frame edges rather than truncated, which keeps the box
 * size fixed. A clamped box reads the same border pixel several times and writes
 * it back several times — again idempotent.
 *
 * ## Binning, and what it does and does not bound
 *
 * `bin_factor = b` replaces each `b x b` background block with its mean, so the
 * background branch carries `ceil(nx/b) * ceil(ny/b) * nz` values. **Binning is a
 * resolution reduction, not an error bound.** With `b > 1` the background
 * reconstruction error is the binning error plus the quantization error, and it
 * is *not* bounded by the background branch's error bound. Only `b = 1` gives a
 * background that genuinely satisfies its stated bound pixel-wise.
 *
 * Both are supported on purpose, and they answer different questions:
 *   - `b = 1` — a true dual-error-bound pipeline; per-region bound verification
 *     is meaningful and must pass on both regions.
 *   - `b > 1` — the ROIBIN configuration; higher ratio, but background fidelity
 *     may only be reported as a distortion metric (PSNR), never as a satisfied
 *     error bound.
 * The ROI branch satisfies its bound in both cases; that is the invariant the
 * science depends on.
 *
 * ## Ports
 *
 * Forward (1 → 3):
 *   - input  0: the field, `float` or `double`, `nx*ny*nz` elements
 *   - output `roi`   : `npeaks * (2*hw+1)^2` elements, same type
 *   - output `bg`    : `ceil(nx/b) * ceil(ny/b) * nz` elements, same type
 *   - output `peaks` : `npeaks * 8` bytes, the peak record table (UINT8)
 * Inverse (3 → 1): the three above, in that order, back to the field.
 *
 * ## Serialized config header
 *   uint32 nx, ny, nz, npeaks; uint16 hw, bin; uint8 dtype, reserved
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

/// One Bragg-peak record, matching the on-disk `.roi` layout exactly (8 bytes).
struct RoiPeak {
    uint32_t z;   ///< frame index (slowest axis)
    uint16_t x;   ///< fast-axis pixel coordinate
    uint16_t y;   ///< slow-axis pixel coordinate
};
static_assert(sizeof(RoiPeak) == 8, "RoiPeak must be 8 bytes to match the .roi format");

/**
 * Region-of-interest / background split stage.
 *
 * `TData` is the field element type (`float` or `double`).
 */
template <typename TData>
class ROIBinSplitStage : public Stage {
public:
    ROIBinSplitStage() = default;
    ~ROIBinSplitStage() override = default;

    // ── Configuration ───────────────────────────────────────────────────────

    /// Load the peak list from a `.roi` file (compress side only). Throws on a
    /// malformed file or on peaks that fall outside the configured dimensions.
    void setPeaksFile(const std::string& path);

    /// Supply peaks directly (used by tests).
    void setPeaks(const std::vector<RoiPeak>& peaks);

    /// ROI box half-width in pixels; the box is `(2*hw+1)^2`. Default 4 → 9x9.
    void setRoiHalfWidth(uint32_t hw) { half_width_ = hw; }
    uint32_t getRoiHalfWidth() const { return half_width_; }

    /// Background binning factor; 1 disables binning. See the header note on
    /// what binning does and does not bound.
    void setBinFactor(uint32_t b) {
        if (b == 0) throw std::runtime_error("ROIBinSplit: bin_factor must be >= 1");
        bin_ = b;
    }
    uint32_t getBinFactor() const { return bin_; }

    /**
     * Pipeline-driven dims push, at addStage and again at finalize.
     *
     * Ignored once the dims came from the archive (deserializeHeader) and ignored
     * for a degenerate push. On the decompress path the pipeline is rebuilt from
     * the FZM header and its global dims are not repopulated, so finalize() pushes
     * {0,0,1} to every stage; taking that would erase the geometry this stage just
     * read from its own header and leave the inverse pass unable to place a single
     * ROI box.
     */
    void setDims(const std::array<size_t, 3>& dims) override {
        if (dims_from_header_ || dims[0] == 0) return;
        dims_ = dims;
    }
    void setDims(size_t x, size_t y = 1, size_t z = 1) {
        setDims(std::array<size_t, 3>{x, y, z});
    }

    size_t getNumPeaks() const { return peaks_.size(); }
    size_t getBoxArea()  const { return size_t(2 * half_width_ + 1) * (2 * half_width_ + 1); }
    size_t getBgNx()     const { return (dims_[0] + bin_ - 1) / bin_; }
    size_t getBgNy()     const { return (dims_[1] + bin_ - 1) / bin_; }
    size_t getBgCount()  const { return getBgNx() * getBgNy() * dims_[2]; }
    size_t getRoiCount() const { return peaks_.size() * getBoxArea(); }

    /// Fraction of ROI slots that are duplicates of an already-covered pixel.
    /// Computed on the host at finalize; reported as a run note so the
    /// redundancy the design trades for an exact output size stays visible.
    double getRoiOverlapFraction() const { return overlap_frac_; }

    // ── Stage control ───────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    /// Plain kernel launches and a stream-ordered H2D upload of the peak table
    /// that happens once at finalize, not per execute → capturable.
    bool isGraphCompatible() const override { return true; }

    // ── Port model ──────────────────────────────────────────────────────────
    size_t getNumInputs()  const override { return is_inverse_ ? 3 : 1; }
    size_t getNumOutputs() const override { return is_inverse_ ? 1 : 3; }

    std::vector<std::string> getOutputNames() const override {
        return is_inverse_ ? std::vector<std::string>{"output"}
                           : std::vector<std::string>{"roi", "bg", "peaks"};
    }

    std::string getName() const override { return "ROIBinSplit"; }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::ROIBIN_SPLIT);
    }

    uint8_t getOutputDataType(size_t output_index) const override {
        if (is_inverse_) return elemType();
        switch (output_index) {
            case 0: return elemType();                             // roi
            case 1: return elemType();                             // bg
            default: return static_cast<uint8_t>(DataType::UINT8);  // peaks
        }
    }
    uint8_t getInputDataType(size_t input_index) const override {
        if (!is_inverse_) return elemType();
        switch (input_index) {
            case 0: return elemType();
            case 1: return elemType();
            default: return static_cast<uint8_t>(DataType::UINT8);
        }
    }

    // ── Execution ───────────────────────────────────────────────────────────
    void execute(
        fz::stream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    /// Uploads the peak table to a stage-private persistent device buffer so the
    /// forward pass never does a per-execute H2D copy.
    void onFinalize(size_t estimated_inlen, MemoryPool* pool) override;

    size_t estimateDeviceFootprintBytes(size_t /*inlen*/) const override {
        return peaks_.size() * sizeof(RoiPeak);
    }

    // ── Size estimation ─────────────────────────────────────────────────────
    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (is_inverse_) {
            return {dims_[0] * dims_[1] * dims_[2] * sizeof(TData)};
        }
        (void)input_sizes;
        return {getRoiCount() * sizeof(TData),
                getBgCount()  * sizeof(TData),
                peaks_.size() * sizeof(RoiPeak)};
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        if (is_inverse_)
            return {{"output", dims_[0] * dims_[1] * dims_[2] * sizeof(TData)}};
        return {{"roi",   getRoiCount() * sizeof(TData)},
                {"bg",    getBgCount()  * sizeof(TData)},
                {"peaks", peaks_.size() * sizeof(RoiPeak)}};
    }

    size_t getActualOutputSize(int index) const override {
        if (is_inverse_) return (index == 0)
            ? dims_[0] * dims_[1] * dims_[2] * sizeof(TData) : 0;
        switch (index) {
            case 0:  return getRoiCount() * sizeof(TData);
            case 1:  return getBgCount()  * sizeof(TData);
            case 2:  return peaks_.size() * sizeof(RoiPeak);
            default: return 0;
        }
    }

    std::vector<std::string> getRunNotes() const override;

    // ── Serialization ───────────────────────────────────────────────────────
    size_t serializeHeader(size_t, uint8_t* buf, size_t max_size) const override {
        if (max_size < kHeaderSize) return 0;
        size_t off = 0;
        auto put32 = [&](uint32_t v) { std::memcpy(buf + off, &v, 4); off += 4; };
        auto put16 = [&](uint16_t v) { std::memcpy(buf + off, &v, 2); off += 2; };
        put32(static_cast<uint32_t>(dims_[0]));
        put32(static_cast<uint32_t>(dims_[1]));
        put32(static_cast<uint32_t>(dims_[2]));
        put32(static_cast<uint32_t>(peaks_.size()));
        put16(static_cast<uint16_t>(half_width_));
        put16(static_cast<uint16_t>(bin_));
        buf[off++] = elemType();
        buf[off++] = 0;  // reserved
        return off;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size < kHeaderSize) return;
        size_t off = 0;
        auto get32 = [&]() { uint32_t v; std::memcpy(&v, buf + off, 4); off += 4; return v; };
        auto get16 = [&]() { uint16_t v; std::memcpy(&v, buf + off, 2); off += 2; return v; };
        dims_[0] = get32(); dims_[1] = get32(); dims_[2] = get32();
        dims_from_header_ = true;
        const uint32_t npeaks = get32();
        half_width_ = get16();
        bin_        = get16();
        // The peak *values* arrive on the `peaks` input port, not in this header;
        // only the count is needed here so the sizes line up before execute().
        peaks_.assign(npeaks, RoiPeak{0, 0, 0});
    }

    size_t getMaxHeaderSize(size_t) const override { return kHeaderSize; }

    void saveState() override {
        saved_dims_ = dims_; saved_hw_ = half_width_;
        saved_bin_ = bin_;   saved_npeaks_ = peaks_.size();
        state_saved_ = true;
    }
    /// Restoring is a no-op until saveState() has actually run. decompressMulti()
    /// brackets each inverse execute() with save/restore, but the pipeline may call
    /// restore first; without this guard that would write the default {0,0,1} over
    /// the geometry deserializeHeader() just recovered and the inverse pass would
    /// fail with "dimensions not set".
    void restoreState() override {
        if (!state_saved_) return;
        dims_ = saved_dims_; half_width_ = saved_hw_; bin_ = saved_bin_;
        if (peaks_.size() != saved_npeaks_) peaks_.resize(saved_npeaks_);
    }

private:
    // 4 x uint32 (dims, npeaks) + 2 x uint16 (hw, bin) + dtype + reserved.
    static constexpr size_t kHeaderSize = 22;

    static constexpr uint8_t elemType() {
        return static_cast<uint8_t>(sizeof(TData) == 4 ? DataType::FLOAT32
                                                       : DataType::FLOAT64);
    }

    /// Host-side check of how many ROI slots land on an already-covered pixel.
    void computeOverlapFraction();

    bool  is_inverse_ = false;
    bool  dims_from_header_ = false;  ///< dims came from the archive; ignore pipeline pushes
    bool  state_saved_      = false;  ///< saveState() has run at least once
    std::array<size_t, 3> dims_ = {0, 0, 1};
    uint32_t half_width_ = 4;
    uint32_t bin_        = 1;
    double   overlap_frac_ = 0.0;

    std::vector<RoiPeak> peaks_;
    RoiPeak* d_peaks_ = nullptr;   ///< pool-owned persistent copy (compress side)

    std::array<size_t, 3> saved_dims_ = {0, 0, 1};
    uint32_t saved_hw_  = 4;
    uint32_t saved_bin_ = 1;
    size_t   saved_npeaks_ = 0;
};

extern template class ROIBinSplitStage<float>;
extern template class ROIBinSplitStage<double>;

} // namespace fz
