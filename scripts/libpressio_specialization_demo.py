#!/usr/bin/env python3
"""Rehearse staged/specialized compression, optionally with SDRBench from Hugging Face."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-path', type=Path, help='Optional installed binding directory')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--input', type=Path, help='Finite, flat f32 input file')
    source.add_argument('--sdrbench', action='store_true', help='Load CESM-ATM/2d CLDHGH from Hugging Face')
    source.add_argument('--decode', type=Path, help='Decode a saved archive in this process')
    parser.add_argument('--cache', type=Path, default=Path(os.environ.get('FZGM_DEMO_CACHE', Path.home()/'.cache/fzgm-demo')))
    parser.add_argument('--elements', type=int, default=262147)
    parser.add_argument('--error-bound', type=float, default=2**-10)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--archive-dir', type=Path, help='Save archives and verify decoding in fresh processes')
    parser.add_argument('--output', type=Path, help='JSON report, or restored .npy for --decode')
    args = parser.parse_args()
    if args.python_path:
        sys.path.insert(0, str(args.python_path.resolve()))
    import numpy as np
    import libpressio_fzgpumodules as fzgm

    if args.decode:
        if not args.output:
            parser.error('--decode requires --output for the restored .npy array')
        restored = fzgm.Compressor().decode(args.decode.read_bytes())
        np.save(args.output, restored)
        print(json.dumps({'decoded': str(args.decode), 'shape': list(restored.shape), 'dtype': str(restored.dtype)}))
        return
    if args.elements <= 0 or args.repeats <= 0 or not np.isfinite(args.error_bound) or args.error_bound <= 0:
        parser.error('elements, repeats, and a finite error bound must be positive')
    provenance = {'source': 'synthetic'}
    if args.sdrbench:
        import sdrbench
        field = sdrbench.dataset('cesm-atm', '2d').field('CLDHGH')
        data = np.ascontiguousarray(field.load(cache=args.cache, source='hf'))
        if data.dtype != np.float32:
            raise RuntimeError(f'Expected float32 SDRBench field, got {data.dtype}')
        digest = hashlib.sha256(data.tobytes()).hexdigest()
        if digest != field.sha256:
            raise RuntimeError('SDRBench field checksum does not match its catalog')
        provenance = {'source': 'huggingface', 'dataset': 'cesm-atm', 'variant': '2d',
                      'field': 'CLDHGH', 'sha256': digest, 'cache': str(args.cache)}
    elif args.input:
        data = np.fromfile(args.input, dtype=np.float32)
        provenance = {'source': 'file', 'path': str(args.input)}
    else:
        data = (np.sin(np.arange(args.elements, dtype=np.float64)/1000) +
                0.3*np.cos(np.arange(args.elements, dtype=np.float64)/37)).astype(np.float32)
    if not data.size or not np.isfinite(data).all():
        parser.error('input must contain finite f32 values')

    def check(restored):
        if restored.shape != data.shape or restored.dtype != data.dtype:
            raise RuntimeError('Decoded dtype or shape changed')
        error = float(np.abs(data.astype(np.float64)-restored.astype(np.float64)).max())
        if not np.isfinite(restored).all() or error > args.error_bound:
            raise RuntimeError(f'Max error {error} exceeds bound {args.error_bound}')
        return error

    results = []
    for policy in ('off', 'auto'):
        compressor = (fzgm.Chain()
            .add('quantizer:float:uint32', linear_mode=True)
            .add('lorenzo:int32', from_port='codes', block_size=32)
            .add('adaptive_bitpack:int32', block_size=32)
            .configure(eb=args.error_bound, memory_strategy='preallocate', specialization=policy))
        elapsed = []
        max_error = 0.0
        for iteration in range(args.repeats+1):
            start = time.perf_counter()
            compressed = compressor.encode(data)
            restored = np.asarray(compressor.decode(compressed))
            duration_ms = (time.perf_counter()-start)*1000
            max_error = max(max_error, check(restored))
            if iteration:
                elapsed.append(duration_ms)
            else:
                cold_ms = duration_ms
        metrics = compressor.get_metrics()
        forward = int(metrics['fzgpumodules:specialization_forward_groups'])
        inverse = int(metrics['fzgpumodules:specialization_inverse_groups'])
        expected = 1 if policy == 'auto' else 0
        if forward != expected or inverse != expected:
            raise RuntimeError(f'{policy}: unexpected specialization counts {forward}/{inverse}')
        fresh_process = False
        if args.archive_dir:
            args.archive_dir.mkdir(parents=True, exist_ok=True)
            archive = args.archive_dir/f'{policy}.fzlp'
            decoded = args.archive_dir/f'{policy}.npy'
            archive.write_bytes(bytes(compressed))
            command = [sys.executable, str(Path(__file__).resolve()), '--decode', str(archive), '--output', str(decoded)]
            if args.python_path:
                command.extend(['--python-path', str(args.python_path.resolve())])
            subprocess.run(command, check=True)
            max_error = max(max_error, check(np.load(decoded, allow_pickle=False)))
            fresh_process = True
        results.append({
            'policy': policy, 'elements': int(data.size), 'shape': list(data.shape),
            'max_abs_error': max_error, 'compression_ratio': data.nbytes/compressed.nbytes,
            'archive_bytes': int(compressed.nbytes),
            'forward_groups': forward, 'inverse_groups': inverse,
            'forward_implementations': metrics['fzgpumodules:specialization_forward_implementations'],
            'inverse_implementations': metrics['fzgpumodules:specialization_inverse_implementations'],
            'fallback_reason': metrics['fzgpumodules:specialization_fallback_reason'],
            'cold_python_roundtrip_ms': cold_ms, 'median_python_roundtrip_ms': median(elapsed),
            'fresh_process_decode_verified': fresh_process,
        })
    report = {'input': provenance, 'absolute_error_bound': args.error_bound, 'results': results,
              'timing_scope': 'Python round trip including host/device transfers; one cold iteration excluded from median',
              'pipeline_scope': 'Block-reset 1-D predictor with 32-element blocks, independent of array shape'}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
