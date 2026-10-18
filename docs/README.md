# Project documentation

Start here, then follow the links.

| Document | What it is |
|---|---|
| [`../README.md`](../README.md) | The project: what it does, how to run it, and the current results. |
| [`superpowers/specs/2026-10-18-indian-regulatory-rag-design.md`](superpowers/specs/2026-10-18-indian-regulatory-rag-design.md) | The design. What the system must do, and why each choice was made. The binding authority. |
| [`superpowers/plans/2026-10-18-corpus-and-retrieval-baseline.md`](superpowers/plans/2026-10-18-corpus-and-retrieval-baseline.md) | The implementation plan for the current build. Task by task, with exact interfaces and tests. |
| [`decisions.md`](decisions.md) | Decision log. What was decided, why, what was rejected, and what it costs if wrong. |
| [`architecture-changelog.md`](architecture-changelog.md) | Append-only log of changes to the system's shape. |
| [`glossary.md`](glossary.md) | Indian legal and regulatory vocabulary, plus retrieval and evaluation vocabulary. |
| [`diagrams/`](diagrams/) | Architecture, data-flow, sequence and module diagrams as standalone HTML. Start with `01-system-architecture.html`. |

## Reading order

For the engineering: `decisions.md`, then the spec, then the current `results.md`.

For the domain: `glossary.md` first. The corpus is Indian tax and banking regulation, and the
vocabulary is dense with acronyms that carry real meaning.

## Scope note

The design describes more than is built. `architecture-changelog.md` records the difference at every
point, and the diagrams mark not-yet-built components explicitly. Nothing here claims a capability
the code does not have.
