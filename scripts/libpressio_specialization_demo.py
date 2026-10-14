#!/usr/bin/env python3
"""Compare staged and specialized compression through libpressio's Python API."""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-path', type=Path, help='Directory containing libpressio.py and _pressio.so')
    parser.add_argument('--input', type=Path, help='Optional finite, flat f32 input file')
    parser.add_argument('--elements', type=int, default=262147)
    parser.add_argument('--error-bound', type=float, default=2**-10)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.python_path:
        sys.path.insert(0, str(args.python_path.resolve()))
    import numpy as np
    lp = importlib.import_module('libpressio')
    Chain = importlib.import_module('fzgm_helper').Chain
    if args.elements <= 0 or args.repeats <= 0 or not np.isfinite(args.error_bound) or args.error_bound <= 0:
        parser.error('elements, repeats, and a finite error bound must be positive')
    data = (np.fromfile(args.input, dtype=np.float32) if args.input else
            (np.sin(np.arange(args.elements, dtype=np.float64) / 1000) +
             0.3 * np.cos(np.arange(args.elements, dtype=np.float64) / 37)).astype(np.float32))
    if not data.size or not np.isfinite(data).all():
        parser.error('input must contain finite f32 values')

    results = []
    for policy in ('off', 'auto'):
        compressor = (Chain()
            .add('quantizer:float:uint32', linear_mode=True)
            .add('lorenzo:int32', from_port='codes', block_size=32)
            .add('adaptive_bitpack:int32', block_size=32)
            .configure(memory_strategy='preallocate', specialization=policy)
            .compressor(eb=args.error_bound))
        elapsed = []
        max_error = 0.0
        # First iteration includes pipeline construction and JIT; exclude it from timing.
        for iteration in range(args.repeats + 1):
            start = time.perf_counter()
            compressed = compressor.encode(data)
            restored = np.asarray(compressor.decode(compressed, np.empty_like(data)))
            duration_ms = (time.perf_counter() - start) * 1000
            if restored.shape != data.shape:
                raise RuntimeError(f'{policy}: logical shape changed')
            error = float(np.abs(data.astype(np.float64) - restored.astype(np.float64)).max())
            if not np.isfinite(restored).all() or error > args.error_bound:
                raise RuntimeError(f'{policy}: max error {error} exceeds bound {args.error_bound}')
            max_error = max(max_error, error)
            if iteration:
                elapsed.append(duration_ms)
        metrics = compressor.get_metrics()
        forward = int(metrics['fzgpumodules:specialization_forward_groups'])
        inverse = int(metrics['fzgpumodules:specialization_inverse_groups'])
        if (policy == 'off' and (forward or inverse)) or (policy == 'auto' and not (forward and inverse)):
            raise RuntimeError(f'{policy}: unexpected specialization counts {forward}/{inverse}')
        results.append({
            'policy': policy, 'elements': int(data.size), 'max_abs_error': max_error,
            'compression_ratio': float(metrics['size:compression_ratio']),
            'forward_groups': forward, 'inverse_groups': inverse,
            'forward_implementations': metrics['fzgpumodules:specialization_forward_implementations'],
            'inverse_implementations': metrics['fzgpumodules:specialization_inverse_implementations'],
            'fallback_reason': metrics['fzgpumodules:specialization_fallback_reason'],
            'median_python_roundtrip_ms': median(elapsed),
        })
    print(json.dumps(results, indent=2))
    print('Timing includes Python and host/device transfers; it is not kernel-only throughput.')


if __name__ == '__main__':
    main()
