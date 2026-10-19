#!/usr/bin/env python3
"""Run isolated graph/staged compression probes; retain raw results and failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from statistics import median

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--executable', required=True, type=Path)
parser.add_argument('--input', required=True, type=Path, help='Raw float32 field')
parser.add_argument('--output-dir', required=True, type=Path)
parser.add_argument('--batches', type=int, default=3)
parser.add_argument('--repeats', type=int, default=30)
args = parser.parse_args()
if args.batches < 1 or args.repeats < 1:
    parser.error('batches and repeats must be positive')
args.output_dir.mkdir(parents=True, exist_ok=True)
x = np.fromfile(args.input, dtype=np.float32)
if not x.size or not np.isfinite(x).all():
    parser.error('input must contain finite float32 data')
sizes = sorted(set(min(n, x.size) for n in (1024, 16384, 262147, x.size)))
for n in sizes:
    np.ascontiguousarray(x[np.linspace(0, x.size-1, n, dtype=np.int64)]).tofile(args.output_dir/f'input-{n}.f32')
cells = [(n, family, mode) for n in sizes for family in ('lorenzo', 'lorenzo_rle', 'demo')
         for mode in ('minimal', 'preallocate', 'graph', 'auto')]
records = []
for batch in range(args.batches):
    order = list(cells)
    random.Random(1049+batch).shuffle(order)
    for n, family, mode in order:
        command = [str(args.executable.resolve()), str((args.output_dir/f'input-{n}.f32').resolve()),
                   str(n), family, mode, str(args.repeats)]
        log = args.output_dir/f'{batch}-{n}-{family}-{mode}.log'
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=90, env=dict({k: v for k, v in os.environ.items() if k not in ('FZ_SPECIALIZE', 'FZ_FUSION')}, CUDA_VISIBLE_DEVICES='0'))
        log.write_text(result.stdout)
        lines = [line[7:] for line in result.stdout.splitlines() if line.startswith('RESULT ')]
        record = {'batch': batch, 'elements': n, 'pipeline': family, 'mode': mode,
                  'returncode': result.returncode, 'log': str(log), 'status': 'failed'}
        if result.returncode == 0 and len(lines) == 1:
            record.update(json.loads(lines[0]), status='ok')
        else:
            record['error'] = result.stdout[-2000:]
        records.append(record)
        (args.output_dir/'raw.json').write_text(json.dumps(records, indent=2)+'\n')
    print(f'Completed batch {batch+1}/{args.batches}: {len(records)} cases', flush=True)
summary = []
for n, family, mode in cells:
    rows = [r for r in records if (r['elements'],r['pipeline'],r['mode']) == (n,family,mode)]
    valid = [r for r in rows if r['status'] == 'ok']
    row = {'elements': n, 'pipeline': family, 'mode': mode, 'valid_batches': len(valid)}
    if valid:
        for key in ('host_ms','gpu_ms','setup_ms','peak_pool_bytes','archive_bytes','max_abs_error','specialized_groups'):
            row[key] = median(r[key] for r in valid)
        row['host_batch_medians_ms'] = [r['host_ms'] for r in valid]
        row['reconstruction_hashes'] = sorted(set(r['reconstruction_hash'] for r in valid))
        row['changed_input_verified'] = all(r['changed_input_verified'] for r in valid)
    summary.append(row)
report = {'input_sha256': hashlib.sha256(x.tobytes()).hexdigest(), 'source_elements': int(x.size),
          'gpu': subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
          'repetitions_per_batch': args.repeats, 'batches': args.batches,
          'scope': 'GPU-resident compression on one H100; stream-synchronized host latency includes CUDA event recording; GPU interval includes queued work and dispatch gaps; setup separate; no graph decompression',
          'summary': summary}
(args.output_dir/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(f'Validated {sum(r["status"] == "ok" for r in records)}/{len(records)} cases')
