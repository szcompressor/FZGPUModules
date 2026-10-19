#!/usr/bin/env python3
"""Verify the installed v2 Python demo and its explicitly selected Spack dependencies."""
import argparse
import ctypes
import importlib.metadata
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--environment-root', required=True, type=Path)
parser.add_argument('--spack-view', required=True, type=Path)
args = parser.parse_args()
root = args.environment_root.resolve()
view = args.spack_view

import numpy as np
import libpressio as lp
import _pressio
import libpressio_fzgpumodules as fzgm
import sdrbench

if not Path(sys.prefix).resolve().is_relative_to(root):
    raise RuntimeError(f'Python is outside demo environment: {sys.prefix}')
site = view/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
origins = {'numpy': np.__file__, 'libpressio_python': lp.__file__,
           'libpressio_extension': _pressio.__file__, 'fzgm_wrapper': fzgm.__file__,
           'sdrbench': sdrbench.__file__}
expected = {'numpy': site/'numpy/__init__.py', 'libpressio_python': site/'libpressio.py',
            'libpressio_extension': site/'_pressio.so', 'fzgm_wrapper': site/'libpressio_fzgpumodules.py'}
for label, path in expected.items():
    if Path(origins[label]).resolve() != path.resolve():
        raise RuntimeError(f'{label} differs from selected Spack installation: {origins[label]}')
if not Path(sdrbench.__file__).resolve().is_relative_to(root):
    raise RuntimeError('SDRBench is outside the demo environment')
libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if '/libfzgmod_' in line or '/liblibpressio.so' in line})
if not any('/libfzgmod_pipeline.so' in path for path in libraries):
    raise RuntimeError('FZGPUModules pipeline library not loaded')
for library in libraries:
    if Path(library).resolve() != (view/'lib'/Path(library).name).resolve():
        raise RuntimeError(f'Library differs from selected Spack installation: {library}')
cudart = ctypes.CDLL('libcudart.so')
cudart.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
cudart.cudaGetDeviceCount.restype = ctypes.c_int
count = ctypes.c_int()
status = cudart.cudaGetDeviceCount(ctypes.byref(count))
if status or count.value != 1:
    raise RuntimeError(f'Expected one visible CUDA device: status={status}, devices={count.value}')
comp = fzgm.Compressor()
options = comp.get_options()
if 'fzgpumodules:specialization' not in options or any(
        key in options for key in ('fzgpumodules:fusion', 'fzgpumodules:graph_mode')):
    raise RuntimeError('Plugin does not expose the current specialization interface')
print(json.dumps({'status': 'ready', 'python': sys.version.split()[0],
                  'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'numcodecs', 'sdrbench')},
                  'visible_cuda_devices': count.value, 'imports': origins, 'libraries': libraries,
                  'spack_view': str(view)}, indent=2))
