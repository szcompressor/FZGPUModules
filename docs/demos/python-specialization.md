# Python specialization demo

The rehearsed environment uses FZGPUModules **v2.0.1**, libpressio adapter
`6eae906`, and SDRBench `0.1.1`. The adapter and packaging PRs are under review;
this is a pinned integration snapshot, not an upstream libpressio release.
The older v2.0.0 demo environment remains preserved separately.

## Ready environment on this host

```bash
source /home/exouser/demo_environments/fzgm-python-v2.0.1/env.sh
python "$FZGM_DEMO_ROOT/run/libpressio_demo_preflight.py" \
  --environment-root "$FZGM_DEMO_ROOT" --spack-view "$FZGM_SPACK_VIEW"
```

Expect `status: ready`, one visible CUDA device, and imports/native libraries
matching the selected Spack view. The Python virtual environment owns SDRBench
and references the installed Spack NumPy, numcodecs, libpressio, and FZGM packages.
It does not import from a libpressio source/build directory. This environment
references absolute paths; rebuild it on another host instead of moving it.

The ready SDK/libpressio installation is described by
`/home/exouser/release_validation/fzgm-spack/env-python-subclass/spack.lock`.
The SDK is the stable release; the development checkout supplies only demo scripts.

## Python sequence

```python
from pathlib import Path
import numpy as np
import sdrbench
import libpressio_fzgpumodules as fzgm

field = sdrbench.dataset("cesm-atm", "2d").field("CLDHGH")
data = np.ascontiguousarray(field.load(cache=Path("demo-cache"), source="hf"))
compressor = (
    fzgm.Chain()
    .add("quantizer:float:uint32", linear_mode=True)
    .add("lorenzo:int32", from_port="codes", block_size=32)
    .add("adaptive_bitpack:int32", block_size=32)
    .configure(eb=2**-10, specialization="auto", memory_strategy="preallocate")
)
encoded = compressor.encode(data)
restored = compressor.decode(encoded)
assert restored.dtype == data.dtype and restored.shape == data.shape
assert np.isfinite(restored).all()
assert np.abs(data.astype(np.float64)-restored.astype(np.float64)).max() <= 2**-10
print("compression ratio:", data.nbytes/encoded.nbytes)
print(compressor.get_metrics())

Path("cldhgh.fzlp").write_bytes(bytes(encoded))
# In a later session, no encoder chain, dtype, or dimensions are needed:
restored = fzgm.Compressor().decode(Path("cldhgh.fzlp").read_bytes())
```

This is a block-reset **1-D predictor** with matching 32-element blocks, even
when the input array has two dimensions. The archive preserves that array's
original shape. It is not a native cuSZp file-format compatibility claim.

## Rehearsal commands

First run the synthetic input, whose partial final block exercises length handling:

```bash
python "$FZGM_DEMO_ROOT/run/libpressio_specialization_demo.py" \
  --archive-dir "$FZGM_DEMO_ROOT/archives/synthetic"
```

Then run the checksum-verified Hugging Face field:

```bash
python "$FZGM_DEMO_ROOT/run/libpressio_specialization_demo.py" --sdrbench \
  --cache "$FZGM_DEMO_CACHE" \
  --archive-dir "$FZGM_DEMO_ROOT/archives/cldhgh"
```

The scripts compare Off and Auto, check dtype/shape/finiteness and the absolute
error bound, and verify zero versus one installed group in each direction.
`--archive-dir` also saves each archive and invokes a **fresh Python process**
to decode it, then checks the restored array. `--output report.json` retains the
report. Fetch/cache the field before presenting; downloads and checksum work
are outside the reported compression timings.

The first round trip is reported separately, including pipeline setup and JIT.
The steady-state figure is the median of five subsequent Python round trips,
including host/device transfers and output allocation. Validation is outside
the timed region. This is not kernel-only throughput or a guaranteed speedup.
Compression ratio includes all bytes of the self-contained libpressio archive.

## Verified results

The refreshed H100 rehearsal passed synthetic input (262,147 float32 elements)
and CLDHGH (1,800 × 3,600 float32 values). Both policies reconstructed within
`2**-10`; forward/inverse groups were 0/0 for Off and 1/1 for Auto. Each saved
archive also passed fresh-process decoding. Reports are retained in
`/home/exouser/release_validation/fzgm-development-refresh/demo-*.json`.

For a new machine, recreate the Spack installation from the pinned recipe and
lock/dependency choices, create a Python environment that can see that Spack
view, and install `sdrbench==0.1.1`. Select the actual CUDA architecture (`80`
for A100, `90` for H100). The standalone Jetstream cuFile workaround is a local
build setting, not part of the public recipe.

CUDA Graph capture is absent from this Python interface. Evaluate its core C++
implementation separately before deciding whether to retain it in the library.
