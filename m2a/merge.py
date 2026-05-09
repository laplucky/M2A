import os
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM

from .utils import (
    cleanup_memory, get_model_config_attr, get_layer_modules,
)
from .data import prepare_samples_unified
from .constraints import build_constraints_single_layer_unified
from .task_vectors import task_vectors_single_layer_unified
from .solvers_qk import cg_single_head_batched
from .solvers_vo import cg_v, cg_o
from .solvers_ffn import (
    ffn_down_dense_project, ffn_gate_dense_project, ffn_up_dense_project,
    cg_ffn_down, cg_ffn_gate, cg_ffn_up,
)
from .checkpoint import save_checkpoint, load_checkpoint, cleanup_checkpoint
from .metrics import (
    compute_frobenius_norm_layer, compute_cosine_similarity_layer,
    compute_layer_selection_mask, compute_dynamic_alpha,
)


def M2A_merge(
    model_base_path: str,
    model_agent_path: str,
    model_reason_path: str,
    texts_r: List[str],
    tokenizer,
    selected_layers: List[int],
    selected_heads: List[int],
    neigh_radius: int = 5,
    lambda_ridge: float = 1e-4,
    cg_maxit: int = 100,
    cg_tol: float = 1e-5,
    compute_dtype: torch.dtype = torch.float32,
    merge_types: str = "qkvof",
    # QK params
    q_rows_per_text: int = 8,
    k_rows_per_text: int = 8,
    w_q: float = 1.0,
    w_k: float = 1.0,
    # VO params
    v_rows_per_text: int = 4,
    o_rows_per_text: int = 4,
    w_v: float = 1.0,
    w_o: float = 1.0,
    # FFN params
    ffn_rows_per_text: int = 4,
    w_ffn: float = 1.0,
    readout_dirs: int = 2,
    # M2A params
    beta: float = 1.0,
    k_threshold: float = 0.5,
    window_size: int = 3,
    seed: int = 42,
    # Device config
    qk_device: str = "auto",
    vo_device: str = "auto",
    ffn_device: str = "auto",
    use_hooks: bool = True,
    max_seq_len: int = 7168,
    # Output directory for early stats
    output_dir: str = None,
) -> Dict[str, Any]:
    """
    M2A-Merge:

    Returns:
        Dictionary containing merged model and statistics
    """

    print("🚀 Starting M2A-Merge: ")
    print("=" * 80)

    import random
    rng = random.Random(seed)

    # Load models
    print("\n📥 Loading models...")
    model_base = AutoModelForCausalLM.from_pretrained(
        model_base_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    ).eval()

    model_agent = AutoModelForCausalLM.from_pretrained(
        model_agent_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    ).eval()

    model_reason = AutoModelForCausalLM.from_pretrained(
        model_reason_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    ).eval()

    d_model = get_model_config_attr(model_agent.config, 'hidden_size')
    n_heads = get_model_config_attr(model_agent.config, 'num_attention_heads')
    head_dim = d_model // n_heads
    kv_heads = get_model_config_attr(model_agent.config, 'num_key_value_heads', n_heads)

    print(f"📋 Config: d_model={d_model}, n_heads={n_heads}, kv_heads={kv_heads}")

    # Device assignment
    if qk_device == "auto":
        qk_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if vo_device == "auto":
        vo_device = "cuda:1" if torch.cuda.device_count() > 1 else qk_device
    if ffn_device == "auto":
        ffn_device = "cuda:2" if torch.cuda.device_count() > 2 else vo_device

    print(f"🔧 Device mapping: QK={qk_device}, VO={vo_device}, FFN={ffn_device}")

    # Parse merge types
    merge_q = 'q' in merge_types.lower()
    merge_k = 'k' in merge_types.lower()
    merge_v = 'v' in merge_types.lower()
    merge_o = 'o' in merge_types.lower()
    merge_f = 'f' in merge_types.lower()

    print(f"🎯 Merge types: {merge_types.upper()} (Q={merge_q}, K={merge_k}, V={merge_v}, O={merge_o}, F={merge_f})")

    # Check layer compatibility (for Qwen3.5 hybrid architecture)
    print("\n🔍 Checking layer compatibility...")
    compatible_layers = []
    incompatible_layers = []
    for li in selected_layers:
        try:
            attn, mlp = get_layer_modules(model_agent, li)
            if attn is not None:
                compatible_layers.append(li)
            else:
                incompatible_layers.append(li)
        except Exception as e:
            incompatible_layers.append(li)

    if incompatible_layers:
        print(f"⚠️  Warning: {len(incompatible_layers)} layers don't have standard attention projections and will be skipped:")
        print(f"   Incompatible layers: {incompatible_layers}")
        print(f"   (e.g., Qwen3.5 linear_attn layers only support FFN merge)")

    if compatible_layers:
        print(f"✓ {len(compatible_layers)} layers are compatible for full Q/K/V/O merge:")
        print(f"   Compatible layers: {compatible_layers}")
    else:
        print("❌ Error: No compatible layers found! This model may not support the merge operation.")
        raise ValueError("No compatible layers for merging")

    # =========================================================================
    # Phase 1: Extract Task Vectors
    # =========================================================================
    print("\n" + "=" * 80)
    print("Phase 1: Extracting Task Vectors")
    print("=" * 80)

    all_agent_task_vectors = {}
    all_reason_task_vectors = {}

    for li in tqdm(selected_layers, desc="Extracting task vectors"):
        # Agent task vectors: W_agent - W_base
        agent_tv = task_vectors_single_layer_unified(
            model_base, model_agent, li, selected_heads, merge_types, scaling_factor=1.0
        )
        all_agent_task_vectors[li] = agent_tv

        # Reasoning task vectors: W_reason - W_base
        reason_tv = task_vectors_single_layer_unified(
            model_base, model_reason, li, selected_heads, merge_types, scaling_factor=1.0
        )
        all_reason_task_vectors[li] = reason_tv

    # =========================================================================
    # Phase 2: Compute Dynamic Alpha and Similarity
    # =========================================================================
    print("\n" + "=" * 80)
    print("Phase 2: Computing Dynamic Alpha and Similarity Scores")
    print("=" * 80)

    layer_metrics = {}
    similarities = []

    for li in selected_layers:
        # Compute Frobenius norms
        norm_agent = compute_frobenius_norm_layer(all_agent_task_vectors[li])
        norm_reason = compute_frobenius_norm_layer(all_reason_task_vectors[li])

        # Compute dynamic alpha
        alpha_l = compute_dynamic_alpha(norm_agent, norm_reason, beta)

        # Compute cosine similarity
        sim_l = compute_cosine_similarity_layer(
            all_agent_task_vectors[li],
            all_reason_task_vectors[li]
        )

        layer_metrics[li] = {
            "norm_agent": norm_agent,
            "norm_reason": norm_reason,
            "alpha": alpha_l,
            "similarity": sim_l
        }
        similarities.append(sim_l)

    # Compute layer selection mask
    mask, smoothed_sim, mean_sim, std_sim, threshold = compute_layer_selection_mask(
        similarities, k_threshold, window_size
    )

    print(f"\n📊 Similarity Statistics:")
    print(f"  Mean: {mean_sim:.4f}")
    print(f"  Std: {std_sim:.4f}")
    print(f"  Threshold (μ - {k_threshold}σ): {threshold:.4f}")
    print(f"  Layers accepted for merge: {sum(mask)}/{len(mask)}")

    # Store mask in layer_metrics
    for i, li in enumerate(selected_layers):
        layer_metrics[li]["mask"] = mask[i]
        layer_metrics[li]["smoothed_similarity"] = smoothed_sim[i]

    # Print per-layer metrics
    print("\n📋 Per-Layer Metrics:")
    print(f"{'Layer':<8} {'α_l':<10} {'S_l':<10} {'S̃_l':<10} {'M_l':<6} {'||ΔAgent||':<15} {'||ΔReason||':<15}")
    print("-" * 80)
    for li in selected_layers:
        m = layer_metrics[li]
        print(f"{li:<8} {m['alpha']:<10.4f} {m['similarity']:<10.4f} {m['smoothed_similarity']:<10.4f} "
              f"{m['mask']:<6} {m['norm_agent']:<15.2f} {m['norm_reason']:<15.2f}")

    # Prepare early stats (before Phase 3: Null-Space Projection)
    early_stats = {
        "layer_metrics": {str(k): v for k, v in layer_metrics.items()},
        "similarity_stats": {
            "mean": mean_sim,
            "std": std_sim,
            "threshold": threshold,
            "layers_accepted": sum(mask),
            "layers_total": len(mask),
            "window_size": window_size,
            "k_threshold": k_threshold
        },
        "config": {
            "beta": beta,
            "k_threshold": k_threshold,
            "window_size": window_size,
            "merge_types": merge_types,
            "selected_layers": selected_layers,
            "selected_heads": selected_heads
        }
    }

    # Save early statistics (before Phase 3: Null-Space Projection)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        stats_file = os.path.join(output_dir, "M2A_merge_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(early_stats, f, ensure_ascii=False, indent=2, default=str)
        print(f"📊 Saved early statistics: {stats_file}")

    # =========================================================================
    # Phase 3: Null-Space Projection (for accepted layers)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Phase 3: Null-Space Projection (for layers with M_l = 1)")
    print("=" * 80)

    # Checkpoint configuration
    checkpoint_dir = os.path.join(output_dir, "checkpoints") if output_dir is not None else "./checkpoints"

    # Try to load checkpoint
    expected_config = {
        "beta": beta,
        "k_threshold": k_threshold,
        "window_size": window_size,
        "merge_types": merge_types,
        "selected_layers": selected_layers,
        "selected_heads": selected_heads
    }

    checkpoint = load_checkpoint(checkpoint_dir, "phase3", expected_config, selected_layers)
    checkpoint_layer = -1
    checkpoint_processed_layers = []
    checkpoint_projected_vectors = {"qk": {}, "vo": {}, "ffn": {}}

    if checkpoint is not None:
        checkpoint_layer = checkpoint.get("layer", -1)
        checkpoint_processed_layers = checkpoint.get("processed_layers", [])
        checkpoint_data = checkpoint.get("data", {})
        checkpoint_projected_vectors = checkpoint_data.get("projected_vectors", {"qk": {}, "vo": {}, "ffn": {}})
        print(f"📦 Loaded checkpoint up to layer {checkpoint_layer}")
        print(f"   Processed layers: {checkpoint_processed_layers}")
        print(f"   Loaded projections for QK: {len(checkpoint_projected_vectors['qk'])} layers")
        print(f"   Loaded projections for VO: {len(checkpoint_projected_vectors['vo'])} layers")
        print(f"   Loaded projections for FFN: {len(checkpoint_projected_vectors['ffn'])} layers")
    else:
        print("🚀 No checkpoint found, starting from scratch")

    # Prepare samples for constraints
    prepped_samples = prepare_samples_unified(
        texts_r, tokenizer, neigh_radius, merge_types,
        q_rows_per_text, k_rows_per_text, v_rows_per_text, o_rows_per_text, ffn_rows_per_text, rng
    )

    # Load model for constraint construction
    model_R_shared = AutoModelForCausalLM.from_pretrained(
        model_agent_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    ).eval()

    projected_reason_task_vectors = {"qk": {}, "vo": {}, "ffn": {}}

    # Merge checkpoint data
    projected_reason_task_vectors.update(checkpoint_projected_vectors)

    # Track processed layers for checkpoint
    processed_layers = checkpoint_processed_layers.copy()

    for li_idx, li in enumerate(tqdm(selected_layers, desc="Null-space projection")):
        # Skip if this layer was already processed (mask=1 and projection done)
        if li in processed_layers:
            print(f"  ⏭️  Layer {li}: Skipped (already processed in checkpoint)")
            continue

        # Skip if mask is 0 (layer rejected)
        if layer_metrics[li]["mask"] == 0:
            print(f"  ⏭️  Layer {li}: Skipped (M_l=0, format conflict detected)")
            continue

        print(f"\n🔄 Processing Layer {li} (M_l=1, α_l={layer_metrics[li]['alpha']:.4f})")

        # Build constraints
        layer_cons = build_constraints_single_layer_unified(
            model_R_shared, prepped_samples, li, selected_heads, merge_types,
            w_q, w_k, q_rows_per_text, k_rows_per_text,
            w_v, w_o, v_rows_per_text, o_rows_per_text,
            w_ffn, ffn_rows_per_text, readout_dirs,
            qk_device, vo_device, ffn_device, compute_dtype, use_hooks, max_seq_len
        )

        # Get reasoning task vectors for this layer
        reason_tv = all_reason_task_vectors[li]

        # Initialize storage
        if merge_q or merge_k:
            projected_reason_task_vectors["qk"][li] = {}
        if merge_v or merge_o:
            projected_reason_task_vectors["vo"][li] = {}

        # QK projection (per-head)
        if (merge_q or merge_k) and "qk" in layer_cons:
            for h in selected_heads:
                if h not in layer_cons["qk"] or h not in reason_tv["qk"]:
                    continue

                cons_h_qk = layer_cons["qk"][h]
                task_qk = reason_tv["qk"][h]

                task_dQ = task_qk.get("dQ") if merge_q and "dQ" in task_qk else None
                task_dK = task_qk.get("dK") if merge_k and "dK" in task_qk else None

                if task_dQ is not None and task_dK is not None:
                    dQ_proj, dK_proj, cg_info = cg_single_head_batched(
                        cons_h_qk, task_dQ, task_dK, lambda_ridge, cg_maxit, cg_tol,
                        device=qk_device, compute_dtype=compute_dtype
                    )

                    projected_reason_task_vectors["qk"][li][h] = {}
                    if merge_q:
                        projected_reason_task_vectors["qk"][li][h]["dQ_proj"] = dQ_proj.cpu()
                    if merge_k:
                        projected_reason_task_vectors["qk"][li][h]["dK_proj"] = dK_proj.cpu()

        # V projection (per-head)
        if merge_v and "vo" in layer_cons:
            for h in selected_heads:
                if h not in layer_cons["vo"] or h not in reason_tv["vo"]:
                    continue

                cons_h_v = layer_cons["vo"][h]
                if "Xi_v" in cons_h_v and cons_h_v["Xi_v"].numel() > 0:
                    if "dV" in reason_tv["vo"][h]:
                        dV_task = reason_tv["vo"][h]["dV"]
                        dV_proj, info_v = cg_v(cons_h_v, dV_task, lambda_ridge, cg_maxit, cg_tol,
                                             device=vo_device, compute_dtype=compute_dtype)

                        if li not in projected_reason_task_vectors["vo"]:
                            projected_reason_task_vectors["vo"][li] = {}
                        if h not in projected_reason_task_vectors["vo"][li]:
                            projected_reason_task_vectors["vo"][li][h] = {}
                        projected_reason_task_vectors["vo"][li][h]["dV_proj"] = dV_proj.cpu()

        # O projection (per-head)
        if merge_o and "vo" in layer_cons:
            for h in selected_heads:
                if h not in layer_cons["vo"] or h not in reason_tv["vo"]:
                    continue

                cons_h_o = layer_cons["vo"][h]
                if "c_vec" in cons_h_o and cons_h_o["c_vec"].numel() > 0:
                    if "dO" in reason_tv["vo"][h]:
                        dO_task = reason_tv["vo"][h]["dO"]
                        dO_proj, info_o = cg_o(cons_h_o, dO_task, lambda_ridge, cg_maxit, cg_tol,
                                             device=vo_device, compute_dtype=compute_dtype)

                        if li not in projected_reason_task_vectors["vo"]:
                            projected_reason_task_vectors["vo"][li] = {}
                        if h not in projected_reason_task_vectors["vo"][li]:
                            projected_reason_task_vectors["vo"][li][h] = {}
                        projected_reason_task_vectors["vo"][li][h]["dO_proj"] = dO_proj.cpu()

        # FFN projection (per-layer)
        if merge_f and "ffn" in layer_cons and "ffn" in reason_tv:
            ffn_cons = layer_cons["ffn"]
            reason_ffn = reason_tv["ffn"]

            # Skip if ffn_cons is empty or has list values (incompatible layer)
            if not ffn_cons or isinstance(ffn_cons.get("X_gate"), list):
                continue

            projected_reason_task_vectors["ffn"][li] = {}

            # Gate projection
            x_gate = ffn_cons.get("X_gate")
            if "dGate" in reason_ffn and x_gate is not None and torch.is_tensor(x_gate) and x_gate.numel() > 0:
                m_gate = ffn_cons["X_gate"].shape[0]
                dGate_task = reason_ffn["dGate"]

                if m_gate <= 4000:
                    dGate_proj, info_gate = ffn_gate_dense_project(ffn_cons, dGate_task,
                                                                  lam=lambda_ridge,
                                                                  device=ffn_device,
                                                                  compute_dtype=compute_dtype)
                else:
                    dGate_proj, info_gate = cg_ffn_gate(ffn_cons, dGate_task, lambda_ridge,
                                                       cg_maxit, cg_tol,
                                                       device=ffn_device,
                                                       compute_dtype=compute_dtype)

                projected_reason_task_vectors["ffn"][li]["dGate_proj"] = dGate_proj.cpu()

            # Up projection
            x_up = ffn_cons.get("X_up")
            if "dUp" in reason_ffn and x_up is not None and torch.is_tensor(x_up) and x_up.numel() > 0:
                m_up = ffn_cons["X_up"].shape[0]
                dUp_task = reason_ffn["dUp"]

                if m_up <= 4000:
                    dUp_proj, info_up = ffn_up_dense_project(ffn_cons, dUp_task,
                                                            lam=lambda_ridge,
                                                            device=ffn_device,
                                                            compute_dtype=compute_dtype)
                else:
                    dUp_proj, info_up = cg_ffn_up(ffn_cons, dUp_task, lambda_ridge,
                                                 cg_maxit, cg_tol,
                                                 device=ffn_device,
                                                 compute_dtype=compute_dtype)

                projected_reason_task_vectors["ffn"][li]["dUp_proj"] = dUp_proj.cpu()

            # Down projection
            h_tensor = ffn_cons.get("H")
            if "dDown_T" in reason_ffn and h_tensor is not None and torch.is_tensor(h_tensor) and h_tensor.numel() > 0:
                m_down = ffn_cons["H"].shape[0]
                dDown_T_task = reason_ffn["dDown_T"]

                if m_down <= 4000:
                    dDown_T_proj, info_down = ffn_down_dense_project(ffn_cons, dDown_T_task,
                                                                   lam=lambda_ridge,
                                                                   device=ffn_device,
                                                                   compute_dtype=compute_dtype)
                else:
                    dDown_T_proj, info_down = cg_ffn_down(ffn_cons, dDown_T_task, lambda_ridge,
                                                        cg_maxit, cg_tol,
                                                        device=ffn_device,
                                                        compute_dtype=compute_dtype)

                projected_reason_task_vectors["ffn"][li]["dDown_T_proj"] = dDown_T_proj.cpu()

        # Mark this layer as processed
        processed_layers.append(li)

        # Save checkpoint after processing this layer
        checkpoint_data = {
            "projected_vectors": projected_reason_task_vectors
        }
        save_checkpoint(
            checkpoint_dir,
            "phase3",
            li,
            processed_layers.copy(),
            checkpoint_data,
            expected_config
        )

        # Cleanup
        del layer_cons
        cleanup_memory()

    # Free constraint model
    if model_R_shared is not None:
        del model_R_shared
        cleanup_memory()

    # Clean up checkpoint after phase 3 completes successfully
    # cleanup_checkpoint(checkpoint_dir, "phase3")

    # =========================================================================
    # Phase 4: Final Layer-wise Merge
    # =========================================================================
    print("\n" + "=" * 80)
    print("Phase 4: Final Layer-wise Merge")
    print("=" * 80)

    print(f"📋 Merge Strategy:")
    print(f"  • M_l = 0: W_merge = W_agent (100% protect Agent format)")
    print(f"  • M_l = 1: W_merge = W_agent + α_l · ΔW_reason_proj (adaptive merge)")

    merge_stats = {
        "total_params_modified": 0,
        "layers_merged": 0,
        "layers_protected": 0,
        "layer_details": {}
    }

    with torch.no_grad():
        for li in tqdm(selected_layers, desc="Applying M2A-Merge"):
            mask_l = layer_metrics[li]["mask"]
            alpha_l = layer_metrics[li]["alpha"]

            layer_detail = {
                "mask": mask_l,
                "alpha": alpha_l,
                "similarity": layer_metrics[li]["similarity"],
                "params_modified": 0
            }

            if mask_l == 0:
                # Skip merge, keep 100% agent weights
                merge_stats["layers_protected"] += 1
                layer_detail["action"] = "protected"
                merge_stats["layer_details"][li] = layer_detail
                continue

            # Apply merge: W_agent + α_l · ΔW_reason_proj
            merge_stats["layers_merged"] += 1
            layer_detail["action"] = "merged"

            layer_target = model_agent.model.layers[li]

            # QK merge
            if (merge_q or merge_k) and li in projected_reason_task_vectors["qk"]:
                for h in selected_heads:
                    if h not in projected_reason_task_vectors["qk"][li]:
                        continue

                    proj_data = projected_reason_task_vectors["qk"][li][h]

                    # Q
                    if merge_q and "dQ_proj" in proj_data:
                        dQ_proj = proj_data["dQ_proj"]
                        WQ_target = layer_target.self_attn.q_proj.weight.data
                        q_start, q_end = h * head_dim, (h + 1) * head_dim
                        WQ_target[q_start:q_end, :] += (alpha_l * dQ_proj.T).to(WQ_target.device)
                        layer_detail["params_modified"] += dQ_proj.numel()

                    # K
                    if merge_k and "dK_proj" in proj_data:
                        dK_proj = proj_data["dK_proj"]
                        WK_target = layer_target.self_attn.k_proj.weight.data
                        kvh = h % kv_heads
                        k_start, k_end = kvh * head_dim, (kvh + 1) * head_dim
                        WK_target[k_start:k_end, :] += (alpha_l * dK_proj.T).to(WK_target.device)
                        layer_detail["params_modified"] += dK_proj.numel()

            # VO merge
            if (merge_v or merge_o) and li in projected_reason_task_vectors["vo"]:
                for h in selected_heads:
                    if h not in projected_reason_task_vectors["vo"][li]:
                        continue

                    proj_data = projected_reason_task_vectors["vo"][li][h]

                    # V
                    if merge_v and "dV_proj" in proj_data:
                        dV_proj = proj_data["dV_proj"]
                        WV_target = layer_target.self_attn.v_proj.weight.data
                        kvh = h % kv_heads
                        v_rows = slice(kvh * head_dim, (kvh + 1) * head_dim)
                        WV_target[v_rows, :] += (alpha_l * dV_proj.T).to(WV_target.device)
                        layer_detail["params_modified"] += dV_proj.numel()

                    # O
                    if merge_o and "dO_proj" in proj_data:
                        dO_proj = proj_data["dO_proj"]
                        WO_target = layer_target.self_attn.o_proj.weight.data
                        o_cols = slice(h * head_dim, (h + 1) * head_dim)
                        WO_target[:, o_cols] += (alpha_l * dO_proj).to(WO_target.device)
                        layer_detail["params_modified"] += dO_proj.numel()

            # FFN merge
            if merge_f and li in projected_reason_task_vectors["ffn"]:
                proj_data = projected_reason_task_vectors["ffn"][li]

                # Gate
                if "dGate_proj" in proj_data:
                    dGate_proj = proj_data["dGate_proj"]
                    Wg_target = layer_target.mlp.gate_proj.weight.data
                    Wg_target += (alpha_l * dGate_proj).to(Wg_target.device)
                    layer_detail["params_modified"] += dGate_proj.numel()

                # Up
                if "dUp_proj" in proj_data:
                    dUp_proj = proj_data["dUp_proj"]
                    Wu_target = layer_target.mlp.up_proj.weight.data
                    Wu_target += (alpha_l * dUp_proj).to(Wu_target.device)
                    layer_detail["params_modified"] += dUp_proj.numel()

                # Down
                if "dDown_T_proj" in proj_data:
                    dDown_T_proj = proj_data["dDown_T_proj"]
                    Wd_target = layer_target.mlp.down_proj.weight.data
                    Wd_target += (alpha_l * dDown_T_proj.T).to(Wd_target.device)
                    layer_detail["params_modified"] += dDown_T_proj.numel()

            merge_stats["total_params_modified"] += layer_detail["params_modified"]
            merge_stats["layer_details"][li] = layer_detail

    print(f"\n✅ M2A-Merge Complete!")
    print(f"  Layers merged: {merge_stats['layers_merged']}")
    print(f"  Layers protected: {merge_stats['layers_protected']}")
    print(f"  Total params modified: {merge_stats['total_params_modified']:,}")

    return {
        "model": model_agent,
        "layer_metrics": layer_metrics,
        "merge_stats": merge_stats,
        "config": {
            "beta": beta,
            "k_threshold": k_threshold,
            "window_size": window_size,
            "merge_types": merge_types,
            "selected_layers": selected_layers,
            "selected_heads": selected_heads
        }
    }
