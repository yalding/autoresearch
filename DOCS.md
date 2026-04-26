# autoresearch Technical Documentation

This document provides a technical overview of the `autoresearch` project, detailing the architecture, model implementation, and autonomous research methodology.

## 1. System Architecture & Autonomous Loop

The system is designed for **autonomous architectural search**. It operates in a loop where an AI agent (e.g., Gemini, Claude) modifies the training code to optimize a specific metric within a fixed resource budget.

### The Research Loop (from `program.md`)
- **Fixed Budget:** Training is strictly limited to 300 seconds (5 minutes) of wall-clock time. This encourages finding the most "compute-optimal" architecture for the given hardware.
- **Metric:** `val_bpb` (Validation Bits Per Byte). This is a vocabulary-size-independent metric, allowing for fair comparison even if the agent changes the tokenizer or architecture.
- **Decision Logic:** 
    - **Advance:** If `val_bpb` improves (decreases), the change is committed and kept.
    - **Revert:** If `val_bpb` worsens or the script crashes, the code is reset to the last known "good" state.

## 2. Model Architecture (`train.py`)

The model is a highly optimized GPT-style transformer with several modern enhancements:

### Components
- **Attention:** `CausalSelfAttention` integrating Flash Attention 3 via the `kernels` package.
- **Value Embeddings (VE):** Implements a ResFormer-style mechanism where certain layers (determined by `has_ve`) mix in a value embedding with an input-dependent gate.
- **Positional Embeddings:** Rotary Positional Embeddings (RoPE) are applied to the Query and Key vectors.
- **Window Pattern:** Uses a sliding window attention pattern (defaulting to `"SSSL"` - Short, Short, Short, Long) to balance local and global context while reducing computation.
- **Normalisation:** RMSNorm is used for stability.
- **Residual Scalars:** Includes learnable `resid_lambdas` and `x0_lambdas` to scale residual connections and initial skip connections.

### Configuration (`GPTConfig`)
- Default depth: 12 layers.
- Default embedding dimension: 768.
- Default heads: 12.
- Sequence length: 2048.

## 3. Optimization Strategy

The training script uses a custom hybrid optimizer: **`MuonAdamW`**.

- **Muon:** Applied to 2D matrix weights (primarily in Linear layers). Muon uses an orthogonalization-based approach (Newton-Schulz iteration) to maintain the spectral properties of the weights.
- **AdamW:** Used for all other parameters, such as biases, 1D weights (embedding/norm), and the learnable residual scalars.
- **Fusing:** Both Muon and AdamW steps are implemented as fused kernels (via `torch.compile` and custom logic) to maximize GPU throughput.
- **Memory Management:** Uses `PYTORCH_ALLOC_CONF="expandable_segments:True"` and manual GC collection to minimize VRAM fragmentation during rapid iterations.

## 4. Data Pipeline & Evaluation (`prepare.py`)

The data pipeline is fixed to ensure experiments are comparable.

- **Dataset:** Downloads shards from the `climbmix-400b-shuffle` dataset.
- **Tokenization:** A BPE tokenizer with a vocab size of 8192, trained using `rustbpe`.
- **Dataloader:** Implements a best-fit packing strategy to ensure efficient use of the sequence length.
- **Ground Truth Evaluation:** The `evaluate_bpb` function calculates the bits-per-byte loss on a pinned validation shard (`shard_06542`).

## 5. Performance Critical Components

- **Flash Attention 3:** Crucial for maintaining high throughput with large sequence lengths (2048).
- **`torch.compile`:** Extensively used across the model and optimizer to fuse operations.
- **Single-GPU Focus:** The code is optimized for a single high-end NVIDIA GPU (tested on H100).
