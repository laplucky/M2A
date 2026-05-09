import os
import json
import argparse
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from m2a import *  # noqa: F401,F403 — re-export all public symbols for backward compatibility
from m2a.data import read_json_samples
from m2a.utils import ensure_dir, get_model_config_attr
from m2a.pipeline import optimized_layerwise_headwise_nullspace_projection


# ========== Entry point ==========

def main():
    parser = argparse.ArgumentParser(description="Efficient layer-wise head-wise null-space projection merging — supports complete Q/K/V/O/FFN constraints")

    # Basic paths
    parser.add_argument("--base", type=str,
                       default="/opt/data/private/hzhcode/huggingface/models/Qwen/Qwen2.5-7B")
    parser.add_argument("--instruct", type=str,
                       default="/opt/data/private/hzhcode/huggingface/models/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--target", type=str,
                       default="/opt/data/private/hzhcode/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

    # Data & constraint params
    parser.add_argument("--texts_r", type=str, required=True, help="Path to JSON samples")
    parser.add_argument("--max_samples_r", type=int, default=10, help="Max number of samples")
    parser.add_argument("--neigh_radius", type=int, default=2, help="Neighborhood radius around boundary tokens")
    parser.add_argument("--q_rows_per_text", type=int, default=8, help="Rows per text for Q constraints")
    parser.add_argument("--k_rows_per_text", type=int, default=8, help="Rows per text for K constraints")

    # Layer & head config
    parser.add_argument("--layers_tail", type=int, default=2, help="Operate on last N layers")
    parser.add_argument("--heads", type=str, default="all", help="Heads to operate on ('all' or comma-separated indices)")

    # Weights & solvers
    parser.add_argument("--w_q", type=float, default=1.0, help="Weight for Q constraints")
    parser.add_argument("--w_k", type=float, default=1.0, help="Weight for K constraints")
    parser.add_argument("--scaling_factor", type=float, default=1.0, help="Task vector scaling factor")
    parser.add_argument("--lambda_ridge", type=float, default=1e-4, help="Ridge parameter")
    parser.add_argument("--cg_maxit", type=int, default=100, help="Max CG iterations")
    parser.add_argument("--cg_tol", type=float, default=1e-5, help="CG convergence tolerance")

    # Compute config
    parser.add_argument("--compute_precision", type=str, choices=["fp32", "fp64"], default="fp32",
                       help="Compute precision")
    # Multi-device config
    parser.add_argument("--qk_device", type=str, default="auto",
                       help="Device for QK constraints ('auto', 'cpu', 'cuda:0', 'cuda:1', etc.)")
    parser.add_argument("--vo_device", type=str, default="auto",
                       help="Device for VO constraints ('auto', 'cpu', 'cuda:0', 'cuda:1', etc.)")
    parser.add_argument("--ffn_device", type=str, default="auto",
                       help="Device for FFN constraints ('auto', 'cpu', 'cuda:0', 'cuda:1', etc.)")

    # Hook config
    parser.add_argument("--use_hooks", action="store_true", default=True,
                       help="Use hooks to capture precise internal layer features (default: True)")
    parser.add_argument("--no_hooks", action="store_true",
                       help="Disable hooks and use the original feature extraction")
    parser.add_argument("--max_seq_len", type=int, default=5120,
                       help="Max allowed sequence length; samples longer than this are skipped (default: 5120)")

    # Unified parameter selection (e.g., from an ultimate merge script)
    parser.add_argument("--merge_types", type=str, default="qk",
                       help="Merge types: any combination of q/k/v/o/f (e.g., 'qk', 'qkvo', 'qkvof', 'f'; default: qk)")

    parser.add_argument("--v_rows_per_text", type=int, default=4, help="Rows per text for V constraints")
    parser.add_argument("--o_rows_per_text", type=int, default=4, help="Rows per text for O constraints")
    parser.add_argument("--ffn_rows_per_text", type=int, default=4, help="Rows per text for FFN-Down constraints")

    parser.add_argument("--readout_dirs", type=int, default=2, help="Number of readout directions c per head/layer")
    parser.add_argument("--w_v", type=float, default=1.0, help="Weight for V constraints")
    parser.add_argument("--w_o", type=float, default=1.0, help="Weight for O constraints")
    parser.add_argument("--w_ffn", type=float, default=1.0, help="Weight for FFN-Down constraints")

    # Output config
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--save_merged_model", action="store_true", help="Save the merged model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    ensure_dir(args.out_dir)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Compute precision
    compute_dtype = torch.float64 if args.compute_precision == "fp64" else torch.float32

    print("🚀 Efficient layer-wise head-wise null-space projection merging — supports complete Q/K/V/O/FFN constraints")
    print("=" * 70)
    print(f"Base: {args.base}")
    print(f"Instruct: {args.instruct}")
    print(f"Target: {args.target}")
    print(f"Task vector scaling factor: {args.scaling_factor}")
    print(f"Compute precision: {args.compute_precision.upper()}")
    print(f"Devices: QK={args.qk_device}, VO={args.vo_device}, FFN={args.ffn_device}")

    # Hook mode
    use_hooks = args.use_hooks and not args.no_hooks
    print(f"Feature extraction: {'Hook-based (recommended)' if use_hooks else 'Original'}")

    start_time = time.time()

    # Load models (on CPU)
    print("\n📥 Loading models onto CPU...")
    model_base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    ).eval()

    model_instruct = AutoModelForCausalLM.from_pretrained(
        args.instruct, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    ).eval()

    model_target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.target, use_fast=True, trust_remote_code=True)

    # Config
    num_layers = get_model_config_attr(model_target.config, 'num_hidden_layers')
    n_heads = get_model_config_attr(model_target.config, 'num_attention_heads')

    selected_layers = list(range(num_layers - args.layers_tail, num_layers))
    if args.heads == "all":
        selected_heads = list(range(n_heads))
    else:
        selected_heads = [int(x) for x in args.heads.split(",")]

    print(f"📋 Selection:")
    print(f"  Layers: {selected_layers}")
    print(f"  Heads: {len(selected_heads)}/{n_heads}")

    # Read data
    texts_R = read_json_samples(args.texts_r, tokenizer, args.max_samples_r)
    print(f"📊 Number of JSON samples: {len(texts_R)}")

    # Run optimized null-space projection
    print("\n🔬 Running optimized layer-wise head-wise null-space projection...")
    stats = optimized_layerwise_headwise_nullspace_projection(
        model_base, model_instruct, model_target,
        texts_R, tokenizer,
        selected_layers, selected_heads,
        args.neigh_radius, args.lambda_ridge, args.cg_maxit, args.cg_tol,
        args.scaling_factor, compute_dtype,
        # Merge types
        args.merge_types,
        # QK
        args.q_rows_per_text, args.k_rows_per_text, args.w_q, args.w_k,
        # VO
        args.v_rows_per_text, args.o_rows_per_text, args.w_v, args.w_o,
        # FFN
        args.ffn_rows_per_text, args.w_ffn, args.readout_dirs, args.seed,
        # Devices
        args.qk_device, args.vo_device, args.ffn_device,
        # Hooks
        use_hooks
    )

    # Save config & stats
    end_time = time.time()
    config_data = {
        "base": args.base, "instruct": args.instruct, "target": args.target,
        "layers": selected_layers, "heads": selected_heads,
        "compute_precision": args.compute_precision,
        "qk_device": args.qk_device,
        "vo_device": args.vo_device,
        "ffn_device": args.ffn_device,
        "use_hooks": use_hooks,
        "neigh_radius": args.neigh_radius,
        "merge_types": args.merge_types,
        "q_rows_per_text": args.q_rows_per_text, "k_rows_per_text": args.k_rows_per_text,
        "w_q": args.w_q, "w_k": args.w_k,
        "v_rows_per_text": args.v_rows_per_text, "o_rows_per_text": args.o_rows_per_text,
        "w_v": args.w_v, "w_o": args.w_o,
        "ffn_rows_per_text": args.ffn_rows_per_text, "w_ffn": args.w_ffn,
        "readout_dirs": args.readout_dirs,
        "scaling_factor": args.scaling_factor,
        "lambda_ridge": args.lambda_ridge,
        "cg_maxit": args.cg_maxit, "cg_tol": args.cg_tol,
        "runtime_seconds": end_time - start_time,
        "optimization": "layerwise_batched_vectorized_qkvo_ffn",
        "stats": stats
    }

    with open(os.path.join(args.out_dir, "optimized_qkvo_ffn_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    # Save merged model
    if args.save_merged_model:
        out_model = os.path.join(args.out_dir, "merged_qkvo_ffn")
        print(f"💾 Saving merged model to: {out_model}")
        model_target.save_pretrained(out_model)
        tokenizer.save_pretrained(out_model)

    print(f"\n✅ Finished! Elapsed: {end_time - start_time:.1f}s")
    print(f"📁 Output directory: {args.out_dir}")
    print(f"🚀 Improvements: supports complete Q/K/V/O/FFN constraints; constraint building reduces from O(N_text×H_head) to O(N_text); vectorized A/AT computations")


if __name__ == "__main__":
    main()
