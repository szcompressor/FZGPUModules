#!/usr/bin/env python3
"""Check that the Linux Python demo loads its installed packages and CUDA runtime."""
import argparse
import ctypes
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--environment-root', required=True, type=Path)
args = parser.parse_args()
root = args.environment_root.resolve()

import numpy as np
import libpressio as lp
import _pressio

origins = {'python_prefix': sys.prefix, 'numpy': np.__file__,
           'libpressio_python': lp.__file__, 'libpressio_extension': _pressio.__file__}
for label, origin in origins.items():
    if not Path(origin).resolve().is_relative_to(root):
        raise RuntimeError(f'{label} loaded outside demo environment: {origin}')

libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if '/libfzgmod_' in line or '/liblibpressio.so' in line})
if not libraries or not any('/libfzgmod_pipeline.so' in path for path in libraries):
    raise RuntimeError('FZGPUModules libraries not loaded')
for library in libraries:
    if not Path(library).resolve().is_relative_to(root):
        raise RuntimeError(f'Library loaded outside demo environment: {library}')

cudart = ctypes.CDLL('libcudart.so')
cudart.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
cudart.cudaGetDeviceCount.restype = ctypes.c_int
count = ctypes.c_int()
status = cudart.cudaGetDeviceCount(ctypes.byref(count))
if status or count.value < 1:
    raise RuntimeError(f'CUDA device unavailable: status={status}, devices={count.value}')
comp = lp.PressioCompressor.from_config({'compressor_id': 'fzgpumodules'})
options = comp.get_options()
if 'fzgpumodules:specialization' not in options or 'fzgpumodules:fusion' in options:
    raise RuntimeError('Plugin lacks the current specialization option')
print(json.dumps({'status': 'ready', 'python': sys.version.split()[0],
                  'numpy_version': np.__version__, 'visible_cuda_devices': count.value,
                  'imports': origins, 'libraries': libraries}, indent=2))
