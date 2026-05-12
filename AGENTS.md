# AGENTS.md — NeuroCore

> Course project: brain-inspired coreset selection for lightweight VLA robotic-arm action prediction.  
> Repo now contains runnable Python source, Jupyter notebooks, and cached experiment results.

## Project Spec (source of truth)

- `README.md` (Chinese) defines the full assignment: baseline → coreset algorithm → validation.
- Expected deliverables: runnable Python source + a research report with figures and analysis.

## Dataset

- **ALOHA Sim Transfer Cube (Human Demonstrations)** from Hugging Face LeRobot.
- ~50 episodes, ~200 MB, multi-view images + 14-DoF joint actions.
- Load via `datasets` library or `lerobot` toolkit.  
  URL: https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human
- For this project you may reduce to **one camera view + single-arm 7-DoF actions**.

## Expected Code Architecture

1. **Baseline** (`baseline/` or `src/baseline.py`)
   - Randomly sample 10 % of trajectories.
   - Extract visual features with **frozen ResNet-18 or CLIP** (offline).
   - Build regression dataset: `[visual_features, language_instruction] → [7-DoF action]`.
   - Train a lightweight **MLP**; report **MSE**.

2. **Coreset Selection** (`src/coreset/`)
   - Replace random sampling with an automated pruning algorithm.
   - Inspiration from README: continuous action-variance filtering (temporal redundancy) or visual-feature clustering (distributional redundancy).
   - Target: select the "most valuable" 10 % subset.

3. **Validation**
   - Retrain the same MLP on the selected 10 % coreset.
   - Report MSE and compare against the random baseline.

## Repo Layout Conventions

| Path | Purpose |
|------|---------|
| `README.md` | Assignment description (Chinese) — do not edit lightly |
| `docs/papers/` | Reference PDFs (ACT, OpenVLA, data pruning, predictive coding) |
| `docs/neurocore-wiki/` | **Obsidian vault** — project knowledge base |
| `.agents/skills/` | OpenCode skill definitions (managed externally) — **not project source** |
| `opencode.json` | Custom OpenCode commands for wiki operations |
| `.env` | Vault path and wiki config |

## Obsidian Wiki (Knowledge Base)

- Vault location is configured in `.env`: `OBSIDIAN_VAULT_PATH="docs/neurocore-wiki"`.
- Custom OpenCode commands are registered in `opencode.json`:
  - `wiki-ingest`, `wiki-update`, `wiki-query`, `wiki-status`, `wiki-lint`, `cross-linker`, `wiki-rebuild`, `wiki-capture`
- `.manifest.json` in the vault tracks which sources have been ingested and which wiki pages were created/updated.
- When adding new papers or documents, prefer ingesting them through the wiki workflow so they are indexed and cross-linked.

## Toolchain Notes

- `requirements.txt` exists and lists core dependencies (`torch`, `transformers`, `datasets`, `lerobot`, etc.).
- **No CI, lint, or test config exists yet** — set up as needed (recommended: `pytest`, `ruff`/`black`).
- Pre-installed OpenCode skills cover HuggingFace, PyTorch Lightning, scikit-learn, scientific visualization, etc.  
  Use them when relevant, but do not treat skill scripts as project dependencies.

## Common Pitfalls

- Do not train the visual feature extractor — it must stay **frozen** per the spec.
- The dataset is tiny (50 episodes); avoid over-engineering. A simple variance or clustering heuristic is sufficient for the coreset step.
- Language instructions in the ALOHA dataset may be sparse or absent; verify whether language labels exist before assuming a multimodal pipeline.
- When writing the research report, the grading emphasis is on **independent thinking about how "redundancy" is defined**, not just implementation.

## Quick Start Checklist for New Sessions

1. Read `README.md` to confirm the latest assignment requirements.
2. Check `docs/neurocore-wiki/projects/NeuroCore/NeuroCore.md` for distilled context from prior sessions.
3. Source code lives in `src/` and `notebooks/` — inspect those for current implementation state.
4. Install dependencies: `pip install -r requirements.txt` (or `uv pip install -r requirements.txt`).
