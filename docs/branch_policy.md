# Stable main and development {#branch_policy}

`main` contains the supported CUDA catalog and production fixes.
`development` includes the current main branch in its ancestry and preserves experimental modules, backend work, research artifacts,
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

Production fixes land on main, then merge main into development. Resolve conflicts
while retaining development-only files and experimental/backend support; do not
replace the development tree wholesale with main. Cherry-picking equivalent fixes
alone does not synchronize branch ancestry.

Prepare targeted main-bound PRs from a main-based branch containing only the
selected, validated development changes. After promotion, merge main back into
development so it remains zero commits behind main.

Internal `docs/releases/` records remain ignored local files on both branches.
The Python wrapper belongs to the separately installed libpressio adapter.
