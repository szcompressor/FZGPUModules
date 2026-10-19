# Stable main and development {#branch_policy}

`v2.0.0` is the shared release baseline; stable fixes through `v2.0.1` are integrated. `main` contains the supported
CUDA catalog and production fixes. `development` includes that release in its
ancestry and preserves experimental modules, backend work, research artifacts,
profiling programs, cards, design notes, and demo materials.

The initial release integration explicitly restored the experimental tree after
the main cleanup. Future merges from main use this shared v2 baseline; review
conflicts and keep development-only files. Promote bounded, validated feature
commits to main through focused pull requests rather than merging the entire
experimental tree into main.

Experimental archive IDs 36, 37, 39–41, and 43 stay assigned permanently.
Development retains their factories and decoders. Stable main reserves their IDs
but does not implement them. Availability is not a general numerical guarantee.

Create feature branches from development, using its existing checkout:

```bash
cd /home/exouser/FZGPUModules-development
git switch -c feature/<name> development
```

Production fixes should land on main and then be integrated into development.
Cherry-pick bounded fixes when a merge would discard development-only files;
review the resulting patches rather than requiring main to be an ancestor. Internal
`docs/releases/` records remain ignored local files on both branches. The Python
Chain helper belongs to the separately installed libpressio adapter.
