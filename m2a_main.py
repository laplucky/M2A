import os
import json
import argparse
import time

import torch
from transformers import AutoTokenizer

from m2a_merge import (
    ensure_dir, cleanup_memory, print_memory_status,
    PreparedSample, prepare_samples_unified,
    build_constraints_single_layer_unified,
    task_vectors_single_layer_unified,
    cg_single_head_batched, cg_v, cg_o,
    ffn_down_dense_project, ffn_gate_dense_project, ffn_up_dense_project,
    cg_ffn_down, cg_ffn_gate, cg_ffn_up,
    get_layer_modules, get_model_config_attr,
    read_json_samples_recursive,
    M2A_merge,
)


def main():
    parser = argparse.ArgumentParser(description="M2A")

    # Model paths
    parser.add_argument("--base", type=str, required=True, help="Base model path")
    parser.add_argument("--agent", type=str, required=True, help="Agent model path (target)")
    parser.add_argument("--reason", type=str, required=True, help="Reasoning model path (instruct)")

    # Data & constraint params
    parser.add_argument("--texts_r", type=str, required=True, help="Path to JSON sample file")
    parser.add_argument("--max_samples_r", type=int, default=200, help="Max number of samples")
    parser.add_argument("--neigh_radius", type=int, default=5, help="Boundary neighborhood radius")

    # Layer & head config
    parser.add_argument("--layers_start", type=int, default=0, help="Start layer index (inclusive)")
    parser.add_argument("--layers_end", type=int, default=None, help="End layer index (exclusive)")
    parser.add_argument("--heads", type=str, default="all", help="Heads to process ('all' or comma-separated)")

    # Solver params
    parser.add_argument("--lambda_ridge", type=float, default=1e-4, help="Ridge parameter (λ)")
    parser.add_argument("--cg_maxit", type=int, default=100, help="Max CG iterations")
    parser.add_argument("--cg_tol", type=float, default=1e-5, help="CG convergence tolerance")

    # Compute config
    parser.add_argument("--compute_precision", type=str, choices=["fp32", "fp64"], default="fp32")

    # Merge types
    parser.add_argument("--merge_types", type=str, default="qkvof",
                       help="Merge types: combination of q/k/v/o/f")

    # QK params
    parser.add_argument("--q_rows_per_text", type=int, default=8)
    parser.add_argument("--k_rows_per_text", type=int, default=8)
    parser.add_argument("--w_q", type=float, default=1.0)
    parser.add_argument("--w_k", type=float, default=1.0)

    # VO params
    parser.add_argument("--v_rows_per_text", type=int, default=4)
    parser.add_argument("--o_rows_per_text", type=int, default=4)
    parser.add_argument("--w_v", type=float, default=1.0)
    parser.add_argument("--w_o", type=float, default=1.0)

    # FFN params
    parser.add_argument("--ffn_rows_per_text", type=int, default=4)
    parser.add_argument("--readout_dirs", type=int, default=2)
    parser.add_argument("--w_ffn", type=float, default=1.0)

    # M2A-Merge specific params
    parser.add_argument("--beta", type=float, default=1.0,
                       help="Scaling factor for dynamic alpha (default: 1.0)")
    parser.add_argument("--k_threshold", type=float, default=0.5,
                       help="Threshold multiplier for similarity mask (default: 0.5)")
    parser.add_argument("--window_size", type=int, default=3,
                       help="Window size for similarity smoothing (default: 3)")

    # Device config
    parser.add_argument("--qk_device", type=str, default="auto")
    parser.add_argument("--vo_device", type=str, default="auto")
    parser.add_argument("--ffn_device", type=str, default="auto")

    # Hook config
    parser.add_argument("--use_hooks", action="store_true", default=True)
    parser.add_argument("--max_seq_len", type=int, default=15000)

    # Output
    parser.add_argument("--output_dir", type=str, default="./M2A_merge_output")
    parser.add_argument("--model_name", type=str, default="M2A_merged_model")
    parser.add_argument("--seed", type=int, default=40)

    args = parser.parse_args()

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Precision
    compute_dtype = torch.float64 if args.compute_precision == "fp64" else torch.float32

    print("🚀 M2A")
    print("=" * 80)
    print(f"Base: {args.base}")
    print(f"Agent: {args.agent}")
    print(f"Reason: {args.reason}")
    print(f"Output: {args.output_dir}")
    print(f"Beta (α scaling): {args.beta}")
    print(f"K threshold: {args.k_threshold}")
    print(f"Window size: {args.window_size}")
    print("=" * 80)

    start_time = time.time()

    # Load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.agent, use_fast=True, trust_remote_code=True)

    # Read samples
    texts_r = read_json_samples_recursive(args.texts_r, tokenizer, args.max_samples_r)
    print(f"📊 Loaded {len(texts_r)} samples for constraints")

    # Load model config to determine layers
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.agent, trust_remote_code=True)

    # Get layer and head info with Qwen3.5 compatibility
    num_layers = get_model_config_attr(config, 'num_hidden_layers')
    n_heads = get_model_config_attr(config, 'num_attention_heads')

    # Layer selection
    if args.layers_end is None:
        args.layers_end = num_layers
    selected_layers = list(range(args.layers_start, args.layers_end))

    # Head selection
    if args.heads == "all":
        selected_heads = list(range(n_heads))
    else:
        selected_heads = [int(x) for x in args.heads.split(",")]

    print(f"📋 Processing:")
    print(f"  Layers: {selected_layers}")
    print(f"  Heads: {len(selected_heads)}/{n_heads}")

    # Run M2A-Merge
    result = M2A_merge(
        args.base, args.agent, args.reason,
        texts_r, tokenizer,
        selected_layers, selected_heads,
        args.neigh_radius, args.lambda_ridge, args.cg_maxit, args.cg_tol,
        compute_dtype, args.merge_types,
        args.q_rows_per_text, args.k_rows_per_text, args.w_q, args.w_k,
        args.v_rows_per_text, args.o_rows_per_text, args.w_v, args.w_o,
        args.ffn_rows_per_text, args.w_ffn, args.readout_dirs,
        args.beta, args.k_threshold, args.window_size, args.seed,
        args.qk_device, args.vo_device, args.ffn_device,
        args.use_hooks, args.max_seq_len,
        args.output_dir,  # Pass output_dir to save early stats
    )

    # Save merged model
    model_output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"\n💾 Saving merged model: {model_output_dir}")
    result["model"].save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    # Save statistics
    stats_file = os.path.join(args.output_dir, "M2A_merge_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        # Convert layer_metrics to serializable format
        serializable_stats = {
            "layer_metrics": {str(k): v for k, v in result["layer_metrics"].items()},
            "merge_stats": result["merge_stats"],
            "config": result["config"]
        }
        json.dump(serializable_stats, f, ensure_ascii=False, indent=2, default=str)

    print(f"📊 Saved statistics: {stats_file}")

    end_time = time.time()
    print(f"\n✅ M2A-Merge completed! Elapsed: {end_time - start_time:.1f}s")
    print(f"📁 Merged model: {model_output_dir}")


if __name__ == "__main__":
    main()
