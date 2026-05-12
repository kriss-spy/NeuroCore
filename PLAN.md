# NeuroCore: Execution Plan

## 0. Environment & Tooling Setup

**Virtual Environment:**

```bash
uv venv .venv
source .venv/bin/activate
```

**Dependencies to install:**

```
torch torchvision transformers datasets scikit-learn matplotlib seaborn pandas numpy tqdm jupyter ipykernel pillow
```

**Register kernel for VS Code:**

```bash
python -m ipykernel install --user --name=neurocore --display-name "Python (neurocore)"
```

**In VS Code:** Select the `neurocore` kernel for all notebooks.

**Git hygiene for notebooks:**

- Before every commit, **clear all cell outputs** (VS Code: "Clear All Outputs").
- Only code + markdown goes into git. This keeps diffs readable and avoids 50MB `.ipynb` files.
- A final executed version will be generated at the end for submission.

---

## 1. Project Directory Structure

```
NeuroCore/
├── .venv/                       # uv virtual environment (gitignored)
├── src/
│   ├── __init__.py
│   ├── data_utils.py            # Dataset loading, episode inspection, action extraction
│   ├── feature_extractor.py     # Frozen ResNet-18 feature extraction + caching
│   ├── baseline.py              # Random episode sampling + MLP training
│   ├── coreset/
│   │   ├── __init__.py
│   │   ├── temporal_filter.py   # Action variance / predictive coding scoring
│   │   ├── distributional_filter.py  # K-means clustering / RAS scoring
│   │   └── select.py            # Unified coreset selection (combines both mechanisms)
│   └── validate.py              # Retrain + evaluate on identical test set
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_coreset_selection.ipynb
│   ├── 05_validation.ipynb
│   └── NeuroCore_full.ipynb     # Final consolidated narrative notebook
├── results/
│   ├── features_resnet18.pt     # Cached visual features
│   ├── checkpoints/
│   │   ├── baseline.pt
│   │   └── coreset.pt
│   ├── figures/                 # All PNG/PDF plots
│   ├── coreset_selection.json   # IDs of selected episodes
│   ├── baseline_metrics.json
│   ├── coreset_metrics.json
│   └── report.md                # Final research report
├── requirements.txt
└── README.md                    # Updated with run instructions
```

---

## 2. Development Workflow: Scripts ↔ Notebook

**Rule:** All reusable logic lives in `src/`. Notebooks import and call `src/` modules.

**Why:** Copilot assists better in `.py` files for complex logic, and notebooks stay clean as narrative documents. You can debug a script in isolation, then call it from a notebook cell.

**Example pattern:**

```python
# In notebook cell
from src.baseline import run_baseline
metrics = run_baseline(random_seed=42)
print(metrics)
```

---

## 3. Phase-by-Phase Execution

### Phase 1: Data Understanding (`notebooks/01_data_exploration.ipynb`)

**Goal:** Load the dataset, understand its anatomy, and make simplification decisions.

**Tasks:**

- Load `lerobot/aloha_sim_transfer_cube_human` via `datasets`.
- Inspect episode count (confirm 50).
- List camera views; select **one** (e.g., `top` or `camera_0`).
- Inspect action tensor shape per frame; confirm 14-DoF.
- Isolate single-arm 7-DoF (first 7 dimensions).
- Check for language instructions. If sparse/null, document this and design baseline without language.
- Plot: episode length distribution, action value distributions, sample frames from an episode.

**Decision checkpoint:** Confirm single-view + single-arm is viable. If language is absent, proceed with visual-only features.

**Output:** A notebook section proving the data is loaded and understood.

---

### Phase 2: Feature Extraction (`src/feature_extractor.py` + `notebooks/02_feature_extraction.ipynb`)

**Goal:** Extract frozen ResNet-18 features for every frame, cache them.

**Implementation (`src/feature_extractor.py`):**

- Load `torchvision.models.resnet18(pretrained=True)`.
- Remove final FC layer to get 512-D feature vectors.
- Set `eval()` mode; freeze all parameters.
- Process frames: resize → normalize (ImageNet stats) → forward pass.
- Extract features for all frames across all 50 episodes.
- Save as `results/features_resnet18.pt` (dict mapping `(episode_idx, frame_idx)` → 512-D tensor).

**Notebook (`02_feature_extraction.ipynb`):**

- Run extraction (one-time cost).
- Visualize feature space with **UMAP** or **t-SNE**, colored by episode.
- Check: do episodes cluster by visual similarity? This informs the distributional filter later.

**Output:** Cached feature file + visualization.

---

### Phase 3: Baseline (`src/baseline.py` + `notebooks/03_baseline.ipynb`)

**Goal:** Randomly sample 5 episodes (10%), train MLP, report MSE.

**Implementation (`src/baseline.py`):**

```python
def run_baseline(random_seed=42):
    # 1. Randomly select 5 episode indices from 0-49
    # 2. Load cached features for those episodes
    # 3. Build dataset: X = features, y = 7-DoF actions
    # 4. Test set: all frames from remaining 45 episodes
    # 5. MLP: 512 → 256 → 128 → 7
    # 6. Train with MSE loss, Adam, track loss curves
    # 7. Evaluate on test set, return MSE + per-joint MSE
```

**Hyperparameters (document these):**

- Learning rate: 1e-3
- Batch size: 32
- Epochs: 50 (or early stopping)
- Hidden dims: [256, 128]

**Notebook (`03_baseline.ipynb`):**

- Call `run_baseline()`.
- Plot training/validation loss curves.
- Report final MSE.
- Save model to `results/checkpoints/baseline.pt`.
- Save metrics to `results/baseline_metrics.json`.

**Output:** Trained baseline model + MSE benchmark.

---

### Phase 4: Coreset Selection (`src/coreset/` + `notebooks/04_coreset_selection.ipynb`)

This is the **heart of the project** and where your independent thinking matters most.

#### 4A. Temporal Filter — Predictive Coding (`src/coreset/temporal_filter.py`)

**Concept:** The brain ignores predictable sensory input. High prediction error = high value.

**Implementation:**

- For each episode, compute per-frame action change: `delta_t = ||a_t - a_{t-1}||_2`.
- Compute episode-level temporal score: mean or max of deltas, or proportion of "non-idle" frames (delta > threshold).
- Episodes with high motion complexity score higher.

**Parameter to document:** Window size (1 vs. 3 frames), threshold for "idle".

#### 4B. Distributional Filter — RAS (`src/coreset/distributional_filter.py`)

**Concept:** The brain filters noise and focuses on high-information-utility moments. We operationalize this as coverage of the action/visual distribution.

**Implementation:**

- Cluster all frame features (from all 50 episodes) using **k-means** (e.g., k=10 or k=20).
- For each episode, compute its cluster occupancy histogram.
- Diversity score = entropy of histogram, or number of unique clusters covered.
- Rare cluster bonus: if an episode is the sole contributor to a cluster, give it extra weight.

**Parameter to document:** k for k-means, distance metric.

#### 4C. Unified Selection (`src/coreset/select.py`)

**Implementation:**

```python
def select_coreset(k=5, alpha=0.5):
    # alpha: weight for temporal score (0=only diversity, 1=only motion)
    # 1. Compute temporal_score for each episode
    # 2. Compute diversity_score for each episode
    # 3. Normalize both to [0, 1]
    # 4. final_score = alpha * temporal_score + (1-alpha) * diversity_score
    # 5. Select top-k episodes
    # 6. Return episode indices + scores
```

**Notebook (`04_coreset_selection.ipynb`):**

- Visualize temporal variance per episode (bar plot).
- Visualize cluster coverage matrix (heatmap: episodes × clusters).
- Show t-SNE/UMAP scatter: highlight selected episodes vs. random baseline episodes.
- Experiment with `alpha` values and document the choice (default: 0.5).
- Save selected IDs to `results/coreset_selection.json`.

**Critical for report:** Document exactly how you defined "redundancy" and why this definition is brain-inspired.

**Output:** Selected 5 episode IDs + visualizations + design rationale.

---

### Phase 5: Validation (`src/validate.py` + `notebooks/05_validation.ipynb`)

**Goal:** Prove the coreset beats random sampling.

**Implementation (`src/validate.py`):**

```python
def run_validation(selected_episodes):
    # 1. Load cached features for selected episodes
    # 2. Build train dataset
    # 3. Load IDENTICAL test set as baseline (45 remaining episodes)
    # 4. Train same MLP architecture with same hyperparameters
    # 5. Evaluate, return MSE
```

**Notebook (`05_validation.ipynb`):**

- Call `run_validation()` with coreset episodes.
- Plot: Baseline MSE vs. Coreset MSE (bar chart).
- Plot: Learning curves overlaid.
- Plot: Per-joint MSE comparison (grouped bar chart).
- Qualitative analysis: show sample frames from selected vs. discarded episodes.

**Output:** Coreset model checkpoint + validation metrics + comparison figures.

---

### Phase 6: Consolidated Master Notebook (`notebooks/NeuroCore_full.ipynb`)

**Goal:** One notebook that tells the complete story for submission.

**Structure:**

1. **Introduction** — Project motivation, brain inspiration.
2. **Setup** — Imports, paths, seeds.
3. **Dataset** — Load, inspect, statistics ( Phase 1).
4. **Features** — Extract and visualize ( Phase 2).
5. **Baseline** — Random 10%, train, evaluate ( Phase 3).
6. **Coreset Algorithm** — Temporal + distributional filtering, selection ( Phase 4).
7. **Validation** — Retrain, compare, visualize ( Phase 5).
8. **Discussion** — Why it works, limitations, future work.
9. **Conclusion**

**Each section** imports from `src/` and references saved results. The narrative is in markdown cells; the code is minimal orchestration.

**Final step:** Run all cells top-to-bottom in VS Code, then save a copy as `NeuroCore_full_executed.ipynb` for submission. Clear outputs on the clean version in git.

---

### Phase 7: Research Report (`results/report.md`)

Convert the notebook narrative into a formal report:

**Sections:**

1. Abstract
2. Introduction (VLA data redundancy + brain inspiration)
3. Related Work (ACT, OpenVLA, Data Pruning, Predictive Coding)
4. Methodology
   - Dataset
   - Baseline architecture
   - Coreset selection algorithm (detailed formulas + pseudocode)
   - **Definition of redundancy** (grading emphasis — dedicate a subsection)
5. Experiments
   - Baseline results
   - Coreset selection results (with figures)
   - Ablation: temporal-only vs. distributional-only vs. combined
6. Discussion
   - Cognitive interpretation
   - Limitations (50 episodes, CPU, single view)
7. Conclusion
8. References
9. Appendix: Code availability

**Figures to include:** All plots from `results/figures/`, embedded as markdown images.

**Export:** VS Code Markdown PDF extension, or Pandoc.

---

## 4. Documentation & Progress Tracking Strategy

### During Development (Obsidian Wiki)

Use the existing wiki at `docs/neurocore-wiki/` to track thinking:

| Wiki Section | Content |
|--------------|---------|
| `journal/` | Daily log: what was implemented, what failed, next steps. |
| `concepts/` | Definitions: predictive coding, RAS, coreset, redundancy metrics. |
| `references/` | Notes on the 4 assigned papers. |
| `synthesis/` | How papers connect to your design choices. |

### In Code (Docstrings & Comments)

- Every function in `src/` gets a docstring with **Args**, **Returns**, and a one-line brain-inspired motivation if applicable.
- Notebook markdown cells explain *why*, not just *what*.

### For Submission

- `README.md` updated with: install instructions, how to run scripts, notebook overview.
- `requirements.txt` with exact versions.
- `results/report.md` as the formal write-up.

---

## 5. Success Criteria (Checklist)

Before declaring the project done, verify:

- [x] Dataset loads correctly; 50 episodes confirmed.
- [x] ResNet-18 features extracted and cached.
- [x] Baseline MLP trains and reports MSE on held-out test set.
- [x] Coreset algorithm selects exactly 5 episodes.
- [x] Coreset MLP retrains and beats (or meaningfully compares to) baseline.
- [x] All figures generated and saved to `results/figures/`.
- [x] `NeuroCore_full.ipynb` runs end-to-end without errors (smoke-tested; training verified via src/ scripts).
- [x] `report.md` is written and includes the "definition of redundancy" section.
- [x] All code committed to git with clean notebook diffs.

---

## 6. First Action (When You Say Go)

1. ~~Create `requirements.txt`.~~
2. ~~Run `uv pip install -r requirements.txt`.~~
3. ~~Create `src/` and `notebooks/` directories.~~
4. ~~Write `src/data_utils.py` and start `notebooks/01_data_exploration.ipynb`.~~

**Current priority:** Finalize research report (`results/report.md`) and consolidated notebook.
