# M2A: Synergizing Mathematical and Agentic Reasoning in Large Language Models

🎉 **Accepted to NeurIPS 2026 (Poster)!**

**M2A** is a training-free model merging method that integrates mathematical reasoning capabilities into language-model agents while preserving effective multi-turn **think–act–observe** behavior.

M2A identifies agent-behavior-critical subspaces from interaction markers, projects reasoning task vectors into their null spaces, and performs adaptive layer-wise merging.

On **SWE-Bench Verified**, M2A improves an 8B coding agent's resolved rate from **44.0% to 51.2% (+7.2 percentage points)** without additional gradient-based training. Additional experiments demonstrate improvements across model families, agent scaffolds, and non-coding agent tasks.

## ✨ Highlights

- **Behavior-Preserving Reasoning Transfer:** Protect agent-critical reasoning–action transitions during model merging.
- **Training-Free Integration:** No additional SFT or RL is required for the merge itself.
- **Adaptive Layer-Wise Merging:** Combine null-space projection, norm-based coefficient calibration, and a similarity-aware layer mask.
- **Cross-Setting Generalization:** Demonstrated improvements across model families, agent scaffolds, and task domains.

## 📊 Main Results

| Setting | Base Agent | M2A | Improvement |
|---|---:|---:|---:|
| Qwen3-8B / OpenHands / SWE-Bench Verified | 44.0% | **51.2%** | **+7.2 pp** |
| Llama-3.1-8B / OpenHands / SWE-Bench Verified | 17.6% | **23.0%** | +5.4 pp |
| Qwen3-8B / Mini-SWE-Agent / SWE-Bench Verified | 42.6% | **46.4%** | +3.8 pp |
| Qwen3-8B / R2E / SWE-Bench Verified | 46.6% | **49.8%** | +3.2 pp |
| Qwen3-8B / GAIA | 73.33% | **78.79%** | +5.46 pp |

These results demonstrate that M2A effectively synergizes mathematical and agentic reasoning, consistently improving agent performance across diverse settings.

## 🔍 Method Overview

M2A consists of four phases:

1. **Task Vector Construction:** Compute the agent and mathematical reasoning task vectors relative to a shared base model.
2. **Adaptive Layer-Wise Merging:** Calibrate layer-wise merge coefficients using task-vector norms and select compatible layers using cosine similarity.
3. **Behavior-Preserving Null-Space Projection:** Identify agent-critical behavioral subspaces from calibration activations and project reasoning updates into their null spaces.
4. **Merge Write-Back:** Integrate the projected reasoning updates into the agent model.

The final merge rule is:

$$
W_{\mathrm{merge}}^{(l)}
=
W_{\mathrm{agent}}^{(l)}
+
M_l \alpha_l \widetilde{\Delta W}_{\mathrm{reason}}^{(l)}
$$

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 📁 Repository Structure

```text
M2A/
├── m2a/                # Core algorithm implementation
├── m2a_main.py         # CLI entry point
├── m2a_merge.py        # Legacy-compatible entry point
├── run_m2a.sh          # Shell runner
├── requirements.txt    # Python dependencies
└── output/             # Generated model outputs
```

## 🚀 Quick Start

M2A requires three model checkpoints:

| Argument | Description |
|---|---|
| `--base` | Base model |
| `--agent` | Agent model |
| `--reason` | Reasoning model |

All three models must share the same tokenizer and Transformer architecture.

### Option 1: Shell Script (Recommended)

Edit `DEFAULT_BASE`, `DEFAULT_AGENT`, `DEFAULT_REASON`, and `DEFAULT_DATA` in `run_m2a.sh`, then run:

```bash
bash run_m2a.sh
```

Alternatively:

```bash
bash run_m2a.sh <base> <agent> <reason> <output_dir>
```

### Option 2: Python CLI

```bash
python m2a_main.py \
    --base /path/to/base \
    --agent /path/to/agent \
    --reason /path/to/reason \
    --output_dir ./output/M2A \
    --model_name m2a
```

## 📦 Outputs

```text
output_dir/
├── <model_name>/          # Merged model
├── M2A_merge_stats.json   # Layer-wise merging statistics
└── checkpoints/           # Intermediate projection checkpoints
```

## 📖 Citation

If you find M2A useful in your research, please cite our NeurIPS 2026 paper:

```bibtex
@article{wang2026m2a,
  title   = {M2A: Synergizing Mathematical and Agentic Reasoning in Large Language Models},
  author  = {Wang, Junjian and Zhou, Xin and Xu, Qiran and Zhan, Kun},
  journal = {arXiv preprint arXiv:2605.09879},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.09879}
}
```

*The official proceedings citation will be updated upon publication.*
