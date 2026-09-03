# Extending FZGPUModules {#extending}

Everything involved in adding a new stage, teaching it to fuse, and the design
notes behind the stages that did not fit the plain-chain model. These pages used
to sit loose at the top level of the manual; they are collected here so the
top-level navigation stays focused on *using* the library.

## Adding a stage

- \subpage how_to_add_a_stage — the mechanical checklist for a stage that lives
  in this repository (files, `StageType` id, CMake, config, CLI, tests).
- \subpage out_of_tree_stage — building a stage in **your own** repository
  against an installed FZGPUModules, without patching the library. Use this while
  a stage is unpublished or private; fold it back into
  [How to Add a New Stage](\ref how_to_add_a_stage) when it lands upstream.
- \subpage developing_stages_deep_dive — the judgment the checklists do not
  cover: is it a stage at all, what shape, what access pattern, how it connects
  to and is optimized by the rest of the pipeline.

## Specialization

- \subpage pipeline_specialization_internals — the declaration contract a stage
  implements to become eligible for [Pipeline Specialization](\ref pipeline_specialization)
  (fusion + finalize-time optimization).

## Design notes

Working notes for stages whose representation is not a plain linear chain. These
are specifications and scoping documents, not user-facing stage references.

- \subpage szx_conditional_representation — SZx as a conditional per-block
  representation, and why it cannot be decomposed into ordinary stages.
- \subpage experimental_szp — `SZpStage`, the quarantined GPU reference
  compressor, and the supported `szp_composed.toml` chain that replaces it.
