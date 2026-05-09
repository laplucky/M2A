import random
from typing import List, Dict, Any
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM

from .utils import cleanup_memory, get_model_config_attr, get_layer_modules
from .data import prepare_samples_unified
from .constraints import build_constraints_single_layer_unified
from .task_vectors import task_vectors_single_layer_unified
from .solvers_qk import cg_single_head_batched, q_dense_project, k_dense_project
from .solvers_vo import cg_v, cg_o, v_dense_project, o_dense_project
from .solvers_ffn import ffn_down_dense_project, cg_ffn_down


def optimized_layerwise_headwise_nullspace_projection(
    model_base, model_instruct, model_target,
    texts_R: List[str], tokenizer,
    selected_layers: List[int], selected_heads: List[int],
    neigh_radius: int, lambda_ridge: float, cg_maxit: int, cg_tol: float,
    scaling_factor: float = 1.0, compute_dtype: torch.dtype = torch.float32,
    # Unified parameter selection
    merge_types: str = "qkvof",  # e.g., "qk", "qkvo", "qkvof"
    # QK params
    q_rows_per_text: int = 8, k_rows_per_text: int = 8, w_q: float = 1.0, w_k: float = 1.0,
    # VO params
    v_rows_per_text: int = 4, o_rows_per_text: int = 4, w_v: float = 1.0, w_o: float = 1.0,
    # FFN params
    ffn_rows_per_text: int = 4, w_ffn: float = 1.0, readout_dirs: int = 2,
    seed: int = 42,
    # Multi-device config
    qk_device: str = "auto", vo_device: str = "auto", ffn_device: str = "auto",
    # Hook config
    use_hooks: bool = True
) -> Dict[str, Any]:
    """Optimized layer-wise head-wise null-space projection (supports Q/K/V/O/FFN)"""

    print("🚀 Starting optimized layer-wise head-wise null-space projection (Q/K/V/O/FFN)...")
    rng = random.Random(seed)

    d_model = get_model_config_attr(model_target.config, 'hidden_size')
    n_heads = get_model_config_attr(model_target.config, 'num_attention_heads')
    head_dim = d_model // n_heads
    kv_heads = get_model_config_attr(model_target.config, 'num_key_value_heads', n_heads)

    print(f"📋 Config: d_model={d_model}, n_heads={n_heads}, kv_heads={kv_heads}")
    print(f"🔧 Task vector scaling factor: {scaling_factor}")
    print(f"Feature extraction: {'Hook-based (recommended)' if use_hooks else 'Original'}")

    # 1) Preprocess samples (unified)
    prepped_samples = prepare_samples_unified(
        texts_R, tokenizer, neigh_radius, merge_types,
        q_rows_per_text, k_rows_per_text, v_rows_per_text, o_rows_per_text, ffn_rows_per_text, rng
    )

    # Multi-device config
    if qk_device == "auto":
        qk_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if vo_device == "auto":
        vo_device = "cuda:1" if torch.cuda.device_count() > 1 else qk_device
    if ffn_device == "auto":
        ffn_device = "cuda:2" if torch.cuda.device_count() > 2 else vo_device

    # Parse merge types
    merge_q = 'q' in merge_types.lower()
    merge_k = 'k' in merge_types.lower()
    merge_v = 'v' in merge_types.lower()
    merge_o = 'o' in merge_types.lower()
    merge_f = 'f' in merge_types.lower()

    print(f"🔧 Temporarily load target model to GPU (multi-GPU mode)")
    print(f"🔧 Device assignment: QK={qk_device}, VO={vo_device}, FFN={ffn_device}")
    print(f"🎯 Merge types: {merge_types.upper()} (Q={merge_q}, K={merge_k}, V={merge_v}, O={merge_o}, F={merge_f})")

    model_R_temp = AutoModelForCausalLM.from_pretrained(
        model_target.config._name_or_path,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    ).eval()

    total_stats = {
        "total_params_modified": 0,
        "total_norm_q": 0.0,
        "total_norm_k": 0.0,
        "total_norm_v": 0.0,
        "total_norm_o": 0.0,
        "total_norm_ffn": 0.0,
        "total_constraint_residual": 0.0,
        "total_cg_iterations": 0,
        "layer_stats": {}
    }

    # 2) Process by layer (key optimization: do each layer as a whole)
    for li_idx, li in enumerate(tqdm(selected_layers, desc="Optimized per-layer processing")):
        print(f"\n🔄 Processing layer {li} ({li_idx+1}/{len(selected_layers)})")

        # 2a) Unified constraint build (QK/VO/FFN)
        print(f"  📐 Building constraints for layer {li} ({merge_types.upper()})...")
        layer_constraints = build_constraints_single_layer_unified(
            model_R_temp, prepped_samples, li, selected_heads, merge_types,
            w_q, w_k, q_rows_per_text, k_rows_per_text,
            w_v, w_o, v_rows_per_text, o_rows_per_text,
            w_ffn, ffn_rows_per_text, readout_dirs,
            qk_device, vo_device, ffn_device, compute_dtype, use_hooks,
            max_seq_len=args.max_seq_len  # use CLI limit
        )

        # 2b) Unified task vector extraction (QK/VO/FFN)
        print(f"  🎯 Extracting task vectors for layer {li} ({merge_types.upper()})...")
        layer_task_vectors = task_vectors_single_layer_unified(
            model_base, model_instruct, li, selected_heads, merge_types, scaling_factor
            )

        layer_stats = {"heads": {}}

        # 2c) Unified solving & applying (QK/VO/FFN) — default to dense solvers; fallback to CG
        for h in tqdm(selected_heads, desc=f"Per-head solving for layer {li} ({merge_types.upper()})", leave=False):
            head_stat = {
                "constraints_qk": 0, "constraints_v": 0, "constraints_o": 0,
                "norm_q": 0.0, "norm_k": 0.0, "norm_v": 0.0, "norm_o": 0.0,
                "residual_norm_qk": 0.0, "residual_norm_v": 0.0, "residual_norm_o": 0.0,
                "cg_iterations": 0, "params_modified": 0
            }

            # —— Q / K (dense by default; fallback to CG) ——
            if (merge_q or merge_k) and "qk" in layer_constraints and h in layer_constraints["qk"]:
                cons_h_qk = layer_constraints["qk"][h]
                total_constraints_qk = cons_h_qk["Xi_q"].shape[0] + cons_h_qk["Xj_k"].shape[0]
                head_stat["constraints_qk"] = total_constraints_qk

                if total_constraints_qk > 0 and h in layer_task_vectors["qk"]:
                    task_qk = layer_task_vectors["qk"][h]
                    dQ_proj = None
                    dK_proj = None

                    # Q component
                    if merge_q and ("dQ" in task_qk) and (cons_h_qk["Xi_q"].numel() > 0):
                        try:
                            dQ_proj, info_q = q_dense_project(cons_h_qk, task_qk["dQ"], lambda_ridge, device="cpu", compute_dtype=compute_dtype)
                            head_stat["norm_q"] = dQ_proj.norm().item()
                            head_stat["residual_norm_qk"] += info_q["residual_norm"]
                            head_stat["cg_iterations"] += info_q.get("iterations", 1)
                        except RuntimeError:
                            # Fallback to CG
                            dQ_proj, _, info_qk = cg_single_head_batched(
                                {"Xi_q": cons_h_qk["Xi_q"], "kj": cons_h_qk["kj"], "sc_q": cons_h_qk["sc_q"],
                                 "Xj_k": torch.empty(0), "qi": torch.empty(0), "sc_k": torch.empty(0)},
                                task_qk["dQ"], torch.zeros_like(task_qk["dQ"]),
                                lambda_ridge, cg_maxit, cg_tol, device="cpu", compute_dtype=compute_dtype
                            )
                            head_stat["norm_q"] = dQ_proj.norm().item()
                            head_stat["residual_norm_qk"] += info_qk["residual_norm"]
                            head_stat["cg_iterations"] += info_qk["iterations"]

                    # K component
                    if merge_k and ("dK" in task_qk) and (cons_h_qk["Xj_k"].numel() > 0):
                        try:
                            dK_proj, info_k = k_dense_project(cons_h_qk, task_qk["dK"], lambda_ridge, device="cpu", compute_dtype=compute_dtype)
                            head_stat["norm_k"] = dK_proj.norm().item()
                            head_stat["residual_norm_qk"] += info_k["residual_norm"]
                            head_stat["cg_iterations"] += info_k.get("iterations", 1)
                        except RuntimeError:
                            _, dK_proj, info_qk = cg_single_head_batched(
                                {"Xi_q": torch.empty(0), "kj": torch.empty(0), "sc_q": torch.empty(0),
                                 "Xj_k": cons_h_qk["Xj_k"], "qi": cons_h_qk["qi"], "sc_k": cons_h_qk["sc_k"]},
                                torch.zeros_like(task_qk["dK"]), task_qk["dK"],
                                lambda_ridge, cg_maxit, cg_tol, device="cpu", compute_dtype=compute_dtype
                            )
                            head_stat["norm_k"] = dK_proj.norm().item()
                            head_stat["residual_norm_qk"] += info_qk["residual_norm"]
                            head_stat["cg_iterations"] += info_qk["iterations"]

                    # Apply to target model weights
                    layer_target_attn, _ = get_layer_modules(model_target, li)
                    with torch.no_grad():
                        if merge_q and (dQ_proj is not None):
                            WQ_target = layer_target_attn.q_proj.weight.data.to(compute_dtype)
                            q_start, q_end = h * head_dim, (h + 1) * head_dim
                            WQ_target[q_start:q_end, :] += dQ_proj.T.to(WQ_target.device)
                            layer_target_attn.q_proj.weight.data = WQ_target.to(layer_target_attn.q_proj.weight.dtype)
                            head_stat["params_modified"] += dQ_proj.numel()
                        if merge_k and (dK_proj is not None):
                            WK_target = layer_target_attn.k_proj.weight.data.to(compute_dtype)
                            kvh = h % kv_heads
                            k_start, k_end = kvh * head_dim, (kvh + 1) * head_dim
                            WK_target[k_start:k_end, :] += dK_proj.T.to(WK_target.device)
                            layer_target_attn.k_proj.weight.data = WK_target.to(layer_target_attn.k_proj.weight.dtype)
                            head_stat["params_modified"] += dK_proj.numel()

            # —— V (dense by default; fallback to CG) ——
            if merge_v and "vo" in layer_constraints and h in layer_constraints["vo"]:
                cons_h_v = layer_constraints["vo"][h]
                if "Xi_v" in cons_h_v and cons_h_v["Xi_v"].numel() > 0:
                    head_stat["constraints_v"] = cons_h_v["Xi_v"].shape[0]

                    if h in layer_task_vectors["vo"] and "dV" in layer_task_vectors["vo"][h]:
                        dV_task = layer_task_vectors["vo"][h]["dV"]
                        try:
                            dV_proj, info_v = v_dense_project(cons_h_v, dV_task, lambda_ridge, device="cpu", compute_dtype=compute_dtype)
                        except RuntimeError:
                            dV_proj, info_v = cg_v(cons_h_v, dV_task, lambda_ridge, cg_maxit, cg_tol, device="cpu", compute_dtype=compute_dtype)

                        with torch.no_grad():
                            layer_target_attn, _ = get_layer_modules(model_target, li)
                            WV_t = layer_target_attn.v_proj.weight.data.to(compute_dtype)
                            kvh = h % kv_heads
                            v_rows = slice(kvh*head_dim, (kvh+1)*head_dim)
                            WV_t[v_rows, :] += dV_proj.T.to(WV_t.device)
                            layer_target_attn.v_proj.weight.data = WV_t.to(layer_target_attn.v_proj.weight.dtype)

                        head_stat["norm_v"] = dV_proj.norm().item()
                        head_stat["residual_norm_v"] = info_v["residual_norm"]
                        head_stat["cg_iterations"] += info_v.get("iterations", 1)
                        head_stat["params_modified"] += dV_proj.numel()

            # —— O (dense by default; fallback to CG) ——
            if merge_o and "vo" in layer_constraints and h in layer_constraints["vo"]:
                cons_h_o = layer_constraints["vo"][h]
                if "c_vec" in cons_h_o and cons_h_o["c_vec"].numel() > 0:
                    head_stat["constraints_o"] = cons_h_o["c_vec"].shape[0]

                    if h in layer_task_vectors["vo"] and "dO" in layer_task_vectors["vo"][h]:
                        dO_task = layer_task_vectors["vo"][h]["dO"]
                        try:
                            dO_proj, info_o = o_dense_project(cons_h_o, dO_task, lambda_ridge, device="cpu", compute_dtype=compute_dtype)
                        except RuntimeError:
                            dO_proj, info_o = cg_o(cons_h_o, dO_task, lambda_ridge, cg_maxit, cg_tol, device="cpu", compute_dtype=compute_dtype)

                        with torch.no_grad():
                            layer_target_attn, _ = get_layer_modules(model_target, li)
                            WO_t = layer_target_attn.o_proj.weight.data.to(compute_dtype)
                            o_cols = slice(h*head_dim, (h+1)*head_dim)
                            WO_t[:, o_cols] += dO_proj.to(WO_t.device)
                            layer_target_attn.o_proj.weight.data = WO_t.to(layer_target_attn.o_proj.weight.dtype)

                        head_stat["norm_o"] = dO_proj.norm().item()
                        head_stat["residual_norm_o"] = info_o["residual_norm"]
                        head_stat["cg_iterations"] += info_o.get("iterations", 1)
                        head_stat["params_modified"] += dO_proj.numel()

            layer_stats["heads"][h] = head_stat
            total_stats["total_params_modified"] += head_stat["params_modified"]
            total_stats["total_norm_q"] += head_stat["norm_q"]
            total_stats["total_norm_k"] += head_stat["norm_k"]
            total_stats["total_norm_v"] += head_stat["norm_v"]
            total_stats["total_norm_o"] += head_stat["norm_o"]
            total_stats["total_constraint_residual"] += (head_stat["residual_norm_qk"] +
                                                        head_stat["residual_norm_v"] +
                                                        head_stat["residual_norm_o"])
            total_stats["total_cg_iterations"] += head_stat["cg_iterations"]

            print(f"    Head {h}: QK constraints={head_stat['constraints_qk']}, V constraints={head_stat['constraints_v']}, "
                  f"O constraints={head_stat['constraints_o']}")
            print(f"Q norm={head_stat['norm_q']:.4f}, K norm={head_stat['norm_k']:.4f}, V norm={head_stat['norm_v']:.4f}, O norm={head_stat['norm_o']:.4f}")
            print(f"Q residual={head_stat['residual_norm_qk']:.6f}, V residual={head_stat['residual_norm_v']:.6f}, O residual={head_stat['residual_norm_o']:.6f}")

        # Handle FFN-Down once per layer
        if merge_f and "ffn" in layer_constraints and layer_constraints["ffn"].get("H", torch.empty(0)).numel() > 0:
            print(f"  🔧 Handling FFN-Down constraints for layer {li}...")
            ffn_cons = layer_constraints["ffn"]

            dDown_T_proj = None
            info_f = None

            if "ffn" in layer_task_vectors and "dDown_T" in layer_task_vectors["ffn"]:
                dDown_T_task = layer_task_vectors["ffn"]["dDown_T"]
                try:
                    # Default to dense solver
                    dDown_T_proj, info_f = ffn_down_dense_project(ffn_cons, dDown_T_task, lambda_ridge, device="cpu", compute_dtype=compute_dtype)
                except RuntimeError:
                    # Fallback to CG
                    dDown_T_proj, info_f = cg_ffn_down(ffn_cons, dDown_T_task, lambda_ridge, cg_maxit, cg_tol, device="cpu", compute_dtype=compute_dtype)

            if dDown_T_proj is not None and info_f is not None:
                with torch.no_grad():
                    _, layer_target_mlp = get_layer_modules(model_target, li)
                    Wd_t = layer_target_mlp.down_proj.weight.data.to(compute_dtype)  # [d_model, d_ff]
                    Wd_t += dDown_T_proj.T.to(Wd_t.device)  # transpose back
                    layer_target_mlp.down_proj.weight.data = Wd_t.to(layer_target_mlp.down_proj.weight.dtype)

                # FFN stats
                layer_stats["ffn"] = {
                    "constraints": ffn_cons["H"].shape[0],
                    "norm": dDown_T_proj.norm().item(),
                    "residual_norm": info_f["residual_norm"],
                    "cg_iterations": info_f.get("iterations", 1),
                    "params_modified": dDown_T_proj.numel()
                }

                total_stats["total_norm_ffn"] += layer_stats["ffn"]["norm"]
                total_stats["total_constraint_residual"] += layer_stats["ffn"]["residual_norm"]
                total_stats["total_cg_iterations"] += layer_stats["ffn"]["cg_iterations"]
                total_stats["total_params_modified"] += layer_stats["ffn"]["params_modified"]

                print(f"  FFN-Down: constraints={layer_stats['ffn']['constraints']}, "
                      f"norm={layer_stats['ffn']['norm']:.4f}, "
                      f"residual={layer_stats['ffn']['residual_norm']:.6f}")

        total_stats["layer_stats"][li] = layer_stats

    # Cleanup temp model
    del model_R_temp
    cleanup_memory()

    print(f"\n✅ Optimized layer-wise head-wise null-space projection done!")
    print(f"  📊 Totals:")
    print(f"     - Total params modified: {total_stats['total_params_modified']:,}")
    if merge_q:
        print(f"     - Total Q weight change norm: {total_stats['total_norm_q']:.6f}")
    if merge_k:
        print(f"     - Total K weight change norm: {total_stats['total_norm_k']:.6f}")
    if merge_v:
        print(f"     - Total V weight change norm: {total_stats['total_norm_v']:.6f}")
    if merge_o:
        print(f"     - Total O weight change norm: {total_stats['total_norm_o']:.6f}")
    if merge_f:
        print(f"     - Total FFN weight change norm: {total_stats['total_norm_ffn']:.6f}")
    print(f"     - Sum of constraint residuals: {total_stats['total_constraint_residual']:.6f}")
    print(f"     - Total CG iterations: {total_stats['total_cg_iterations']}")

    return total_stats
