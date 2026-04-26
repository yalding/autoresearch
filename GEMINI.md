# Gemini CLI Project Context: autoresearch

## Project Overview
`autoresearch` is an autonomous LLM research system designed to allow AI agents to conduct architectural and hyperparameter experiments independently. The core concept is to provide a fixed 5-minute training budget and a clear metric (`val_bpb` - bits per byte) for comparing different iterations.

### Main Technologies
- **Language:** Python 3.10+
- **Deep Learning:** PyTorch (v2.9.1+cu128)
- **Dependency Management:** [uv](https://docs.astral.sh/uv/)
- **Optimization:** Muon + AdamW
- **Attention:** Flash Attention 3 (via `kernels` package)
- **Tokenization:** BPE (via `rustbpe` and `tiktoken`)

## Core Project Structure
The repository is designed to be minimal, with three primary files:

- **`prepare.py`**: **READ-ONLY for agents.** Contains fixed constants, data preparation logic (downloading shards, training tokenizer), and the ground-truth evaluation harness (`evaluate_bpb`).
- **`train.py`**: **The primary target for agent modification.** Contains the full GPT model architecture, optimizer configuration, and the training loop. Everything in this file is subject to experimental iteration.
- **`program.md`**: **Instructional context for the agent.** Defines the "autonomous research org" rules, experiment loop, and logging requirements. Humans modify this to guide the agents.

## Getting Started

### Prerequisites
- A single NVIDIA GPU (H100 recommended, though smaller GPUs are supported with tuning).
- `uv` project manager installed.

### Key Commands
- **Install Dependencies:**
  ```bash
  uv sync
  ```
- **One-time Data Preparation:**
  ```bash
  uv run prepare.py
  ```
- **Run a Single Training Experiment:**
  ```bash
  uv run train.py
  ```
- **Check Results:**
  ```bash
  grep "^val_bpb:" run.log
  ```

## Development Conventions

### Autonomous Research Loop
1.  **Tagging:** Experiments are conducted on dedicated branches (e.g., `autoresearch/mar25`).
2.  **Modification:** The agent modifies `train.py` with a new research idea.
3.  **Commit:** Every experimental change must be committed to git.
4.  **Execution:** The training script is run for exactly 300 seconds (5 minutes) of wall-clock training time.
5.  **Evaluation:**
    - If `val_bpb` improves (decreases), the change is **kept** and the branch advances.
    - If `val_bpb` is worse or the run crashes, the branch is **reset** to the previous state.
6.  **Logging:** Results are recorded in a tab-separated `results.tsv` (untracked by git).

### Constraints
- **Do not modify `prepare.py`.** It must remain fixed to ensure evaluation consistency.
- **Do not add new dependencies** to `pyproject.toml`.
- **VRAM Usage:** Must stay within the limits of the available GPU (H100 defaults).
- **Simplicity:** Favor code simplicity. Minor improvements that add significant complexity should be discarded.

### Metrics
- **`val_bpb`**: Validation bits per byte. The primary metric for comparison.
- **`peak_vram_mb`**: Monitored to avoid OOM crashes.
- **`mfu_percent`**: Model Flops Utilization.
- **`num_params_M`**: Model size in millions of parameters.
