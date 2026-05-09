from typing import List, Dict, Any

import torch

from .utils import get_model_config_attr, get_layer_modules


# ========== Unified task vector extraction function ==========

def task_vectors_single_layer_unified(
    model_base, model_instruct, layer: int, selected_heads: List[int],
    merge_types: str = "qkvof", scaling_factor: float = 1.0
) -> Dict[str, Any]:
    """Unified single-layer task vector extraction; supports QK/VO/FFN"""
    d_model = get_model_config_attr(model_base.config, 'hidden_size')
    H = get_model_config_attr(model_base.config, 'num_attention_heads')
    hD = d_model // H
    KV = get_model_config_attr(model_base.config, 'num_key_value_heads', H)

    # Parse required processing types
    merge_q = 'q' in merge_types.lower()
    merge_k = 'k' in merge_types.lower()
    merge_v = 'v' in merge_types.lower()
    merge_o = 'o' in merge_types.lower()
    merge_f = 'f' in merge_types.lower()

    print(f"  Extract task vector types: {merge_types.upper()}")

    # Get layer objects with compatibility for different architectures
    attn_base, mlp_base = get_layer_modules(model_base, layer)
    attn_instruct, mlp_instruct = get_layer_modules(model_instruct, layer)

    # Check if this layer has compatible attention (e.g., Qwen3.5 linear_attn returns None)
    if attn_base is None or attn_instruct is None:
        print(f"  ⚠️  Layer {layer} doesn't have standard attention projections (e.g., Qwen3.5 linear_attn), skipping Q/K/V/O merge")
        # Only process FFN if requested
        task_vectors = {"qk": {}, "vo": {}, "ffn": {}}
        if merge_f and mlp_base is not None and mlp_instruct is not None:
            with torch.no_grad():
                dGate = (mlp_instruct.gate_proj.weight - mlp_base.gate_proj.weight) * scaling_factor
                dUp   = (mlp_instruct.up_proj.weight   - mlp_base.up_proj.weight)   * scaling_factor
                dDown = (mlp_instruct.down_proj.weight - mlp_base.down_proj.weight) * scaling_factor
                dDown_T = dDown.T.contiguous()
                task_vectors["ffn"] = {
                    "dGate": dGate.cpu(),
                    "dUp": dUp.cpu(),
                    "dDown_T": dDown_T.cpu()
                }
        return task_vectors

    task_vectors = {"qk": {}, "vo": {}, "ffn": {}}

    with torch.no_grad():
        # QK task vectors
        if merge_q or merge_k:
            dQ = (attn_instruct.q_proj.weight - attn_base.q_proj.weight) * scaling_factor if merge_q else None
            dK = (attn_instruct.k_proj.weight - attn_base.k_proj.weight) * scaling_factor if merge_k else None
        else:
            dQ = dK = None

        # VO task vectors
        if merge_v or merge_o:
            dV = (attn_instruct.v_proj.weight - attn_base.v_proj.weight) * scaling_factor if merge_v else None
            dO = (attn_instruct.o_proj.weight - attn_base.o_proj.weight) * scaling_factor if merge_o else None
        else:
            dV = dO = None

        # FFN task vectors (complete gate/up/down)
        if merge_f:
            dGate = (mlp_instruct.gate_proj.weight - mlp_base.gate_proj.weight) * scaling_factor  # [d_ff, d_model]
            dUp   = (mlp_instruct.up_proj.weight   - mlp_base.up_proj.weight)   * scaling_factor  # [d_ff, d_model]
            dDown = (mlp_instruct.down_proj.weight - mlp_base.down_proj.weight) * scaling_factor  # [d_model, d_ff]
            # Use transposed for Down to match CG implementations
            dDown_T = dDown.T.contiguous()  # [d_ff, d_model]
        else:
            dGate = dUp = dDown_T = None

        # Slice QK task vectors by head
        if merge_q or merge_k:
            for h in selected_heads:
                qk_head = {}

                if merge_q and dQ is not None:
                    q_start, q_end = h * hD, (h + 1) * hD
                    dQ_h = dQ[q_start:q_end, :].T.contiguous()  # [d_model, hD]
                    qk_head["dQ"] = dQ_h.cpu()

                if merge_k and dK is not None:
                    kvh = h % KV
                    k_start, k_end = kvh * hD, (kvh + 1) * hD
                    dK_h = dK[k_start:k_end, :].T.contiguous()  # [d_model, hD]
                    qk_head["dK"] = dK_h.cpu()

                task_vectors["qk"][h] = qk_head

        # Slice VO task vectors by head
        if merge_v or merge_o:
            for h in selected_heads:
                vo_head = {}

                if merge_v and dV is not None:
                    kvh = h % KV
                    v_rows = slice(kvh*hD, (kvh+1)*hD)
                    dV_h = dV[v_rows, :].T.contiguous()  # [d_model, hD]
                    vo_head["dV"] = dV_h.cpu()

                if merge_o and dO is not None:
                    o_cols = slice(h*hD, (h+1)*hD)
                    dO_h = dO[:, o_cols].contiguous()  # [d_model, hD]
                    vo_head["dO"] = dO_h.cpu()

                task_vectors["vo"][h] = vo_head

        # FFN task vectors (not per head)
        if merge_f:
            task_vectors["ffn"] = {}
            if dGate is not None:
                task_vectors["ffn"]["dGate"] = dGate.cpu()
            if dUp is not None:
                task_vectors["ffn"]["dUp"] = dUp.cpu()
            if dDown_T is not None:
                task_vectors["ffn"]["dDown_T"] = dDown_T.cpu()

    return task_vectors
