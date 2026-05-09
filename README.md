# M2A: Synergizing Mathematical and Agentic Reasoning in Large Language Models


## Directory Layout

```
M2A/
├── m2a/              # Core library (algorithm implementation)
├── m2a_main.py       # CLI entry point (argparse)
├── m2a_merge.py      # Legacy-compatible entry point
├── run_m2a.sh        # One-shot runner script
└── output/           # Merged-model outputs
```

## Requirements

```bash
pip install -r requirements.txt
```

## Inputs

| Argument | Description |
|---|---|
| `--base`    | Base model |
| `--agent`   | Agent model |
| `--reason`  | Reasoning model |

All three models must share the same tokenizer and Transformer architecture.

## Quick Start

**Option 1: Shell script (recommended)**

Edit the `DEFAULT_BASE / DEFAULT_AGENT / DEFAULT_REASON / DEFAULT_DATA` paths at the top of `run_m2a.sh`, then:

```bash
bash run_m2a.sh
```

Or override via positional arguments:

```bash
bash run_m2a.sh <base> <agent> <reason> <output_dir>
```

**Option 2: Python CLI**

```bash
python m2a_main.py \
    --base   /path/to/base \
    --agent  /path/to/agent \
    --reason /path/to/reason \
    --output_dir ./output/M2A \
    --model_name m2a \
```

## Internal Pipeline (4 phases)

1. **Task Vectors** — per layer, compute `ΔW_agent` and `ΔW_reason`.
2. **Dynamic α + Layer Mask** — use similarity to decide each layer's merge strength and whether to skip it.
3. **Null-space Projection** — for selected layers, build constraints from calibration activations and project `ΔW_reason` onto the null space of the Agent's behavior. Checkpointing is supported.
4. **Merge Write-back** — `W_merge = W_agent + α_l · ΔW_reason_proj`.

## Output

```
output_dir/
├── <model_name>/            # Merged model (save_pretrained format)
├── M2A_merge_stats.json     # Per-layer α, similarity, mask, and other statistics
└── checkpoints/             # Phase-3 intermediates (used for resume)
```

