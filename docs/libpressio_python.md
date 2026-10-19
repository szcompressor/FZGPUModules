# Python through libpressio

FZGPUModules' native SDK and its libpressio adapter are separate installations.
Build libpressio with `LIBPRESSIO_HAS_FZGPUMODULES=ON` and
`BUILD_PYTHON_WRAPPER=ON` against FZGPUModules 2.0.1. The adapter installs
`libpressio_fzgpumodules.py` alongside libpressio's Python bindings.
Upstream support is under review; the current demo uses pinned libpressio
revision `6eae906` and the stable v2.0.1 SDK.

```python
import libpressio_fzgpumodules as fzgm

compressor = (
    fzgm.Chain()
    .add("quantizer:float:uint32", linear_mode=True)
    .add("lorenzo:int32", from_port="codes", block_size=32)
    .add("adaptive_bitpack:int32", block_size=32)
    .configure(eb=1e-3, specialization="auto", memory_strategy="preallocate")
)
encoded = compressor.encode(data)  # contiguous float32 NumPy array
restored = compressor.decode(encoded)
```

`configure()` ends the builder and returns a subclass of
`libpressio.PressioCompressor`. Options, configuration, and metrics use its
inherited methods. Encoding requires an explicit pipeline; the default
absolute bound is 1e-3 if omitted. `specialization="off"` is the default;
`"auto"` installs eligible implementations and otherwise falls back to staged
execution. Inspect the forward/inverse specialization metrics after a round
trip rather than assuming an implementation was installed or is faster.

The encoded archive contains the pipeline, dtype, and dimensions. A fresh
`fzgm.Compressor().decode(encoded)` can restore it; saved Python `bytes` are
also accepted. Ordinary libpressio factory access remains supported, but its
generic decoder requires an explicit output array. The Python subclass's
output-allocation convenience does not automatically apply to outer native
metacompressors. CUDA Graph capture and experimental SDK stages are not exposed.

The full SDRBench/Hugging Face sequence, environment checks, saved-archive
example, and rehearsal commands are in
[the Python demo runbook](demos/python-specialization.md).
