# Shared Agent Context

This repository is developed with multiple agentic coding tools, including Claude Code
and Codex. Treat their context as shared rather than maintaining a separate Codex-specific
source of truth.

Before working in this repository:

1. Read and follow `CLAUDE.md`. It is the primary repository guidance for every coding
   agent, despite its filename.
2. Read `memory/MEMORY.md` when the task may depend on project history, prior decisions,
   known bugs, plans, user feedback, or ongoing work. Follow its links selectively as
   relevant to the task.
3. Reuse and update the existing shared files (`CLAUDE.md`, `memory/`, documentation,
   plans, and task artifacts) rather than creating parallel Codex-only memory or context.
4. Preserve context written by other agents. Do not rewrite or remove it unless the task
   requires that change and the new information supersedes it.
5. When durable project knowledge emerges, put it in the existing appropriate shared
   file and update `memory/MEMORY.md` when adding a new memory document.

These rules apply regardless of whether the repository is opened locally through WSL or
remotely through VS Code SSH.
