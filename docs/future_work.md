# Engineering future work

This backlog records agreed directions after the IPDPS submission. Items below
are plans, not promises of support or changes to the current stable API.

## Demo and integration

- Finish the Python presentation using the pinned v2.0.1 SDK, libpressio-owned
  wrapper, and cached SDRBench Hugging Face field. Retain correctness,
  specialization-installation, and fresh-process decoding checks.
- Rehearse on the intended Jetstream/A100 environment. Build a dedicated demo
  image if it will be used for the presentation or shared with others; verify a
  newly launched instance from that image, including its data and dependency paths.
- Address reviews of the FZGM plugin, SWIG compatibility, and Spack package PRs.
  After upstream support lands, replace temporary fork pins/branch exclusions and
  retire the packaging patch when the selected source includes its fix.

## Library engineering

- Rank the paper's specialization limitations and existing implementation gaps
  into bounded improvements with correctness, memory, and performance gates.
  Separate legal specialization from input-dependent profitability.
- Continue AMD/backend correctness, coverage, packaging, and performance work on
  development before promoting supported configurations to stable main.
- Reassess experimental modules individually before promoting them; their presence
  on development is not a numerical or performance guarantee.

## Planned CUDA Graph capture deprecation — deferred

The agreed direction is to phase out core CUDA Graph capture rather than expand
its stage support or optimize it for the demo. Implement this in a future work
item; no API is formally deprecated or removed by this backlog entry.

- In a future v2 release, mark the public graph-capture entry points deprecated,
  remove them from recommended workflows, and document compatibility restrictions
  and the changed-input LorenzoQuant/RLE defect.
- Retain existing entry points during the deprecation period; remove graph replay,
  capture-specific stage hooks, buffer handling, and compatibility tests in an
  appropriate breaking release.
- Keep the evaluation and reproducer as engineering records. Small-input gains
  exist, but the tested full-field demo did not benefit and pool memory increased.
  Evidence and limits are in [CN-GRAPH-1](codebase_notes.md#cn-graph-1--cuda-graph-replay-evaluation-after-v201).
- Keep graph capture absent from the Python plugin and demo.

## Research scoping

- Scope automated compressor discovery as a possible next paper: define the
  candidate pipeline space, quality/error constraints, search objectives,
  baselines, and evaluation budget before building a broad search system.
