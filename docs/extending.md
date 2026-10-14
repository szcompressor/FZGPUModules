# Extending FZGPUModules {#extending}

Everything involved in adding a new stage.

## Adding a stage

- \subpage how_to_add_a_stage — the mechanical checklist for a stage that lives
  in this repository (files, `StageType` id, CMake, config, CLI, tests).
- \subpage out_of_tree_stage — building a stage in **your own** repository
  against an installed FZGPUModules, without patching the library. Use this while
  a stage is unpublished or private; fold it back into
  [How to Add a New Stage](\ref how_to_add_a_stage) when it lands upstream.

## Specialization

- \subpage pipeline_specialization_internals — the declaration contract a stage
  implements to become eligible for [Pipeline Specialization](\ref pipeline_specialization)
  (fusion + finalize-time optimization).
