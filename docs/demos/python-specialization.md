# Python specialization demo

This runbook covers the verified Linux/H100 Python demo of the patched
`fzgpumodules` libpressio plugin. The library candidate is v2.0.0; publication
is still pending. The adapter is maintained separately, with a tested installed demo path.

## Ready environment on this host

The installed environment is `/home/exouser/demo_environments/fzgm-python-v2-stable`.
The SDK was freshly built at `66fd9ea`; the final candidate `01f40d5` differs
only in README/building documentation. The adapter is pinned at `b330cbe`.
The prior frozen environment remains available separately.

It contains its own virtual environment, FZGPUModules SDK, installed libpressio,
copied scripts, CLDHGH input, offline Python wheels, logs, and provenance files.
It imports neither the benchmark virtual environment nor the libpressio build
folder. Source the environment file in a dedicated rehearsal shell:

```bash
source /home/exouser/demo_environments/fzgm-python-v2-stable/env.sh
"$FZGM_DEMO_ROOT/venv/bin/python" -I \
  "$FZGM_DEMO_ROOT/run/libpressio_demo_preflight.py" \
  --environment-root "$FZGM_DEMO_ROOT"
```

Expect `status: ready`, one visible CUDA device, no experimental optional-stage
wrappers, an installed `fzgm_helper.Chain`, and import/library paths under
that environment root. `-I` isolates Python from user packages, `PYTHONPATH`,
and the current working directory. The environment file selects GPU 0 and
sets installed library paths plus CUDA 12.9's runtime directory.

Run the small synthetic example, then the real field:

```bash
"$FZGM_DEMO_ROOT/venv/bin/python" -I \
  "$FZGM_DEMO_ROOT/run/libpressio_specialization_demo.py"

"$FZGM_DEMO_ROOT/venv/bin/python" -I \
  "$FZGM_DEMO_ROOT/run/libpressio_specialization_demo.py" \
  --input "$FZGM_DEMO_ROOT/data/CLDHGH.f32"
```

Neither command needs `--python-path`. The scripts require the installed
adapter changes; the unpatched libpressio fork will not reproduce this demo.
The Python virtual environment is tied to its creation path. Rebuild it on
another host instead of moving this directory and assuming it is portable.

## Presentation sequence

1. Show the input and goal: compress a scientific f32 field on one GPU while
   checking the reconstruction against a requested absolute bound.
2. Show the three Python stage tokens: linear Quantizer → integer Lorenzo →
   AdaptiveBitpack. Connect Quantizer's `codes` port and use matching 32-element
   blocks for Lorenzo and AdaptiveBitpack. The file is flattened; this is a
   block-reset 1D pipeline, not a claim of native 2D cuSZp container compatibility.
3. Run Off: show zero installed groups, compression ratio, preserved length,
   and maximum reconstruction error.
4. Run Auto with the same graph: show `warp-register` and
   `warp-register-inverse` installation. Explain that supported combinations
   can replace separate stage execution while retaining the graph.
5. Discuss timing only after the correctness and installation checks. The
   script discards one warmup round trip per policy, including pipeline setup
   and JIT, then reports the median of five repeated Python round trips.
   This includes host/device transfers and Python overhead, not just kernels.

The script performs Off and Auto in one invocation. For live coding, show its
`early_config` dictionary and the single `fzgpumodules:specialization` setting.
The old plugin `fusion` key was removed and produces a migration error.

## Expected checks

The default ABS bound is `2^-10 = 0.0009765625`. This exactly representable bound
keeps the demonstration's strict check easy to inspect. Both directions warm
up before measured repetitions. The script checks finite reconstruction, exact
logical shape, maximum error no greater than the requested bound, and actual
forward/inverse specialization counts.

| Input | Off groups (forward/inverse) | Auto groups | Compression ratio | Max ABS error |
|---|---|---|---|---|
| Synthetic, 262,147 f32 values | 0 / 0 | 1 / 1 | 3.27139 | 0.0009765625 |
| CLDHGH, 6,480,000 f32 values | 0 / 0 | 1 / 1 | 3.86749 | 0.0009765625 |

These values describe the retained candidate and inputs, not a promise for any
field or error bound. Auto is an admission policy, not a per-input performance
prediction. First-use JIT and timing variability should be explained, not hidden.
No general speedup claim is established by this rehearsal.

Encode and decode use the same compressor instance. The plugin does not yet
restore a fresh decoder from a raw compressed buffer. The library's independent
FZM file interface is separate. CUDA Graph mode stays disabled; it is mutually
exclusive with specialization. Keep SPECK/wavelet/SZx experiments out of this
stable-core demonstration.

## Rebuild the environment

Use a dedicated directory outside source checkouts. Set `FZGM_SOURCE` to the
stable library source snapshot, `LIBPRESSIO_SOURCE` to the pinned adapter, and
`FZGM_DEVELOPMENT_SOURCE` to the development checkout containing demo scripts.
The ready environment also includes `rebuild.sh`, captured source snapshots,
and offline wheels for reproduction. The verified versions are
Python 3.12.3, NumPy 2.5.2, numcodecs 0.16.5, typing_extensions 4.16.0, SWIG 4.2.0,
CUDA 12.9, GCC 13.3, and CMake 3.28.3. Exact local versions and patches are recorded
in `manifest.json`. Numcodecs is required by libpressio's high-level wrapper;
NumPy alone is insufficient.

Set the paths for the machine and the **patched** adapter source:

```bash
export FZGM_DEMO_ROOT=/path/to/persistent/fzgm-python-demo
export FZGM_SOURCE=/path/to/FZGPUModules
export LIBPRESSIO_SOURCE=/path/to/patched/libpressio-fork
export FZGM_DEVELOPMENT_SOURCE=/path/to/FZGPUModules-development
export STD_COMPAT_CONFIG=/path/to/libstdcompat/share/std_compat/cmake
export CUDA_ROOT=/usr/local/cuda-12.9

python3 -m venv "$FZGM_DEMO_ROOT/venv"
"$FZGM_DEMO_ROOT/venv/bin/python" -m pip install \
  -r "$FZGM_DEVELOPMENT_SOURCE/python/demo-requirements.txt"
```

`std_compat` 0.0.22 is an installed C++ build dependency. This is a fresh Python
and adapter installation on the existing CUDA host, not an OS-from-scratch
validation. The current manifest records the dependency's Spack installation.
The retained `wheels/` directory allows offline Python installation on the same
Python/platform using `pip install --no-index --find-links /path/to/wheels -r
/path/to/requirements.txt`.

Build/install the SDK if a reviewed installed candidate is not already available.
The architecture below targets the visible GPU. On a GPU-less build host,
specify the target architecture explicitly (`80` for A100, `90` for H100):

```bash
cmake -S "$FZGM_SOURCE" -B "$FZGM_DEMO_ROOT/build-fzgm" \
  -DCMAKE_BUILD_TYPE=Release -DFZGMOD_BACKEND=CUDA \
  -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="$FZGM_DEMO_ROOT/fzgpumodules"
cmake --build "$FZGM_DEMO_ROOT/build-fzgm" -j4
cmake --install "$FZGM_DEMO_ROOT/build-fzgm"
```

Install the native adapter and Python files into their explicit destinations:

```bash
FZGM_PYTHON_SITE=$("$FZGM_DEMO_ROOT/venv/bin/python" -c \
  'import sysconfig; print(sysconfig.get_path("purelib"))')
cmake -S "$LIBPRESSIO_SOURCE" -B "$FZGM_DEMO_ROOT/build-libpressio" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DLIBPRESSIO_BUILD_MODE=CORE -DBUILD_PYTHON_WRAPPER=ON \
  -DLIBPRESSIO_HAS_FZGPUMODULES=ON \
  -DLIBPRESSIO_HAS_SIZE=ON -DLIBPRESSIO_HAS_TIME=ON \
  -DCMAKE_INSTALL_PREFIX="$FZGM_DEMO_ROOT/libpressio" \
  -DPython3_EXECUTABLE="$FZGM_DEMO_ROOT/venv/bin/python" \
  -DLIBPRESSIO_PYTHON_SITELIB="$FZGM_PYTHON_SITE" \
  -DFZGPUModules_DIR="$FZGM_DEMO_ROOT/fzgpumodules/lib/cmake/FZGPUModules" \
  -Dstd_compat_DIR="$STD_COMPAT_CONFIG"
cmake --build "$FZGM_DEMO_ROOT/build-libpressio" -j4
cmake --install "$FZGM_DEMO_ROOT/build-libpressio"

mkdir -p "$FZGM_DEMO_ROOT/run" "$FZGM_DEMO_ROOT/data" "$FZGM_DEMO_ROOT/logs"
cp "$FZGM_DEVELOPMENT_SOURCE/scripts/libpressio_specialization_demo.py" \
   "$FZGM_DEVELOPMENT_SOURCE/scripts/libpressio_demo_preflight.py" "$FZGM_DEMO_ROOT/run/"
cp "$FZGM_SOURCE/data/CLDHGH.f32" "$FZGM_DEMO_ROOT/data/"
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$FZGM_DEMO_ROOT/libpressio/lib:$FZGM_DEMO_ROOT/fzgpumodules/lib:$CUDA_ROOT/lib64"
unset PYTHONPATH FZ_SPECIALIZE FZ_FUSION
```

Libpressio's Python destination is separate from `CMAKE_INSTALL_PREFIX`; set
`LIBPRESSIO_PYTHON_SITELIB` explicitly. Do not assume `pip install libpressio`
provides this fork's patched GPU plugin. After rebuilding, rerun the preflight
and both demonstrations, then record the new revisions, patches, versions,
input hashes, loaded library paths, and outputs.

## Rehearsal and fallback

Before presenting, run preflight, the small example, and CLDHGH in a fresh shell.
Keep the verified environment frozen until release changes require a deliberate
rebuild. Run GPU work serially during rehearsal. An installed-path memcheck is:

```bash
compute-sanitizer --tool memcheck --error-exitcode 42 \
  "$FZGM_DEMO_ROOT/venv/bin/python" -I \
  "$FZGM_DEMO_ROOT/run/libpressio_specialization_demo.py"
```

If setup fails, check the preflight's first error: missing Python dependency,
wrong library paths, unavailable CUDA device, or an unpatched plugin. If Auto
reports zero groups, inspect the graph, numeric mode, block sizes, and any
`FZ_SPECIALIZE` environment override; do not describe staged fallback as
specialized execution. The script deliberately fails that check for this known
eligible demo graph. Clear unexpected overrides before rehearsal.

Keep `logs/synthetic-demo.log` and `logs/cldhgh-demo.log` as clearly labeled
recorded fallback results. They support explaining the workflow if the GPU is
unavailable, without presenting recorded output as a live run.

Validation of the retained installation used isolated Python from the neutral
`run/` directory with the adapter build directory hidden. The installed
Python suite passed 36 checks with one explicit SZx skip, including CUDA-resident
outputs and specialization built through the installed Chain helper. The same
suite and the synthetic demo passed memcheck with zero errors. The configured
source/build paths were hidden throughout the six validation processes. Import and loaded-library provenance passed preflight.
No tag or release was created by this verification.

## Recorded rehearsal

The documented sequence was rehearsed from a fresh shell using the installed
environment: preflight, the synthetic field, then complete CLDHGH. All expected
length, bound, ratio, and installation checks passed. Logs are retained as
`logs/preflight.json`, `logs/synthetic-demo.log`, and
`logs/cldhgh-demo.log`. The recorded CLDHGH median Python round trips were 45.32 ms Off and 42.54 ms
Auto. These host-inclusive timings accompany the workflow and installation
checks; this single rehearsal does not establish a general speedup.
