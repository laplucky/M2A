import math
import random
from typing import List, Dict, Any
from tqdm import tqdm

import torch

from .utils import get_model_config_attr, get_layer_modules
from .data import PreparedSample
from .features import collect_layer_features_with_hooks, compute_sampled_attention_rows_from_qk


# ========== Unified constraint construction function (supports QK/VO/FFN) ==========

def build_constraints_single_layer_unified(
    model_R, prepped_samples: List[PreparedSample],
    layer: int, selected_heads: List[int],
    merge_types: str = "qkvof",
    # QK parameters
    w_q: float = 1.0, w_k: float = 1.0, q_rows_per_text: int = 8, k_rows_per_text: int = 8,
    # VO parameters
    w_v: float = 1.0, w_o: float = 1.0, v_rows_per_text: int = 4, o_rows_per_text: int = 4,
    # FFN parameters
    w_ffn: float = 1.0, ffn_rows_per_text: int = 4, readout_dirs: int = 2,
    # Device configuration
    qk_device: str = "cuda:0", vo_device: str = "cuda:0", ffn_device: str = "cuda:0",
    compute_dtype: torch.dtype = torch.float32, use_hooks: bool = True,
    # Sequence length limit
    max_seq_len: int = 7168
) -> Dict[str, Any]:
    """Unified per-layer constraint construction; supports QK/VO/FFN constraints"""
    device_R = next(model_R.parameters()).device
    d_model = get_model_config_attr(model_R.config, 'hidden_size')
    H = get_model_config_attr(model_R.config, 'num_attention_heads')
    hD = d_model // H
    KV = get_model_config_attr(model_R.config, 'num_key_value_heads', H)
    d_ff = get_model_config_attr(model_R.config, 'intermediate_size', d_model * 4)

    merge_q = 'q' in merge_types.lower()
    merge_k = 'k' in merge_types.lower()
    merge_v = 'v' in merge_types.lower()
    merge_o = 'o' in merge_types.lower()
    merge_f = 'f' in merge_types.lower()

    print(f"  Building constraint types: {merge_types.upper()} (Q={merge_q}, K={merge_k}, V={merge_v}, O={merge_o}, F={merge_f})")

    # Get layer modules with compatibility for different architectures
    attn, mlp = get_layer_modules(model_R, layer)

    # Check if this layer has compatible attention (e.g., Qwen3.5 linear_attn returns None)
    if attn is None:
        print(f"  ⚠️  Layer {layer} doesn't have standard attention projections, skipping Q/K/V/O constraints")
        # Return empty constraints for Q/K/V/O, only process FFN if needed
        constraints = {
            "qk": {},
            "vo": {},
            "ffn": {
                "H": [], "c": [], "sc": [],
                "X_gate": [], "c_gate": [], "sc_gate": [],
                "X_up": [], "c_up": [], "sc_up": []
            } if merge_f else {}
        }
        # Could still build FFN constraints if mlp is available, but for now just return empty
        return constraints

    # weights
    WQ = attn.q_proj.weight.data.clone().to(qk_device) if (merge_q or merge_k) else None
    WK = attn.k_proj.weight.data.clone().to(qk_device) if (merge_q or merge_k) else None
    WV = attn.v_proj.weight.data.clone().to(vo_device) if (merge_v or merge_o) else None
    WO = attn.o_proj.weight.data.clone().to(vo_device) if (merge_v or merge_o) else None

    if merge_f:
        Wg = mlp.gate_proj.weight.data.clone().to(ffn_device)
        Wu = mlp.up_proj.weight.data.clone().to(ffn_device)
        Wd = mlp.down_proj.weight.data.clone().to(ffn_device)
    else:
        Wg = Wu = Wd = None

    constraints = {
        "qk": {h: {
            "Xi_q": [], "kj": [], "sc_q": [],
            "Xj_k": [], "qi": [], "sc_k": []
        } for h in selected_heads} if (merge_q or merge_k) else {},
        "vo": {h: {
            "Xi_v": [], "rv": [], "sc_v": [],
            "c_vec": [], "z_h": [], "sc_o": []
        } for h in selected_heads} if (merge_v or merge_o) else {},
        "ffn": {
            "H": [], "c": [], "sc": [],
            "X_gate": [], "c_gate": [], "sc_gate": [],
            "X_up": [], "c_up": [], "sc_up": []
        } if merge_f else {}
    }

    # random readout directions
    if merge_v or merge_o or merge_f:
        rng = random.Random(1234)
        if merge_v or merge_o:
            C_out_vo = [torch.randn(d_model, device=vo_device).to(next(model_R.parameters()).dtype) for _ in range(readout_dirs)]
            C_out_vo = [c / (c.norm() + 1e-6) for c in C_out_vo]
        else:
            C_out_vo = []

        if merge_f:
            C_out_ffn_gate_up = [torch.randn(d_ff, device=ffn_device).to(next(model_R.parameters()).dtype) for _ in range(readout_dirs)]
            C_out_ffn_gate_up = [c / (c.norm() + 1e-6) for c in C_out_ffn_gate_up]

            C_out_ffn_down = [torch.randn(d_model, device=ffn_device).to(next(model_R.parameters()).dtype) for _ in range(readout_dirs)]
            C_out_ffn_down = [c / (c.norm() + 1e-6) for c in C_out_ffn_down]
        else:
            C_out_ffn_gate_up = []
            C_out_ffn_down = []

    skipped_samples = 0
    for samp in tqdm(prepped_samples, desc=f"Build constraints for layer {layer} ({merge_types.upper()})", leave=False):
        ids = samp.input_ids.to(device_R)

        seq_length = ids.shape[-1]
        if seq_length > max_seq_len:
            skipped_samples += 1
            continue

        if use_hooks:
            layer_features = collect_layer_features_with_hooks(model_R, ids, [layer], merge_types, max_seq_len)
            if layer not in layer_features or not layer_features[layer]:
                continue

            X_attn = layer_features[layer].get("attn_input") if (merge_q or merge_k or merge_v or merge_o) else None
            X_ffn = layer_features[layer].get("ffn_input") if merge_f else None
            q_proj_out = layer_features[layer].get("q_proj_out") if (merge_v or merge_o) else None
            k_proj_out = layer_features[layer].get("k_proj_out") if (merge_v or merge_o) else None

            gate_output = layer_features[layer].get("gate_output") if merge_f else None
            up_output = layer_features[layer].get("up_output") if merge_f else None
        else:
            X_attn = None
            X_ffn = None
            q_proj_out = None
            k_proj_out = None
            gate_output = None
            up_output = None

        T = ids.shape[1]

        # =========================
        # QK constraints
        # =========================
        Q_full = None
        K_full = None
        X_qk = None

        if (merge_q or merge_k) and X_attn is not None:
            X_qk = X_attn.to(qk_device)
            Q_full = X_qk @ WQ.T if WQ is not None else None
            K_full = X_qk @ WK.T if WK is not None else None

        for h in selected_heads:
            if Q_full is not None and K_full is not None:
                Q_h = Q_full[:, h*hD:(h+1)*hD]
                kvh = h % KV
                K_h = K_full[:, kvh*hD:(kvh+1)*hD]

                if merge_q and hasattr(samp, 'pairs_q') and samp.pairs_q:
                    Xi_q = X_qk[[i for i, _ in samp.pairs_q]]
                    kj = K_h[[j for _, j in samp.pairs_q]]
                    sc_q = torch.full((Xi_q.size(0), 1), w_q / math.sqrt(hD))

                    constraints["qk"][h]["Xi_q"].append(Xi_q.cpu())
                    constraints["qk"][h]["kj"].append(kj.cpu())
                    constraints["qk"][h]["sc_q"].append(sc_q)

                if merge_k and hasattr(samp, 'pairs_k') and samp.pairs_k:
                    Xj_k = X_qk[[j for _, j in samp.pairs_k]]
                    qi = Q_h[[i for i, _ in samp.pairs_k]]
                    sc_k = torch.full((Xj_k.size(0), 1), w_k / math.sqrt(hD))

                    constraints["qk"][h]["Xj_k"].append(Xj_k.cpu())
                    constraints["qk"][h]["qi"].append(qi.cpu())
                    constraints["qk"][h]["sc_k"].append(sc_k)

        # =========================
        # VO constraints
        # =========================
        if (merge_v or merge_o) and X_attn is not None:
            X_vo = X_attn.to(vo_device)
            V_full = X_vo @ WV.T if WV is not None else None  # [T, KV*hD]

            sampled_ts = []
            if merge_v and hasattr(samp, "v_t") and samp.v_t:
                sampled_ts.extend([t for t in samp.v_t if t < T])
            if merge_o and hasattr(samp, "o_t") and samp.o_t:
                sampled_ts.extend([t for t in samp.o_t if t < T])

            sampled_rows = None
            if sampled_ts and q_proj_out is not None and k_proj_out is not None:
                sampled_rows = compute_sampled_attention_rows_from_qk(
                    q_out=q_proj_out,
                    k_out=k_proj_out,
                    config=model_R.config,
                    selected_heads=selected_heads,
                    sampled_ts=sampled_ts,
                    device=vo_device,
                )

            for h in selected_heads:
                if V_full is None or sampled_rows is None or h not in sampled_rows:
                    continue

                kvh = h % KV
                V_h = V_full[:, kvh*hD:(kvh+1)*hD]  # [T, hD]

                # V constraint
                if merge_v and hasattr(samp, 'v_t') and samp.v_t:
                    for t in samp.v_t:
                        if t >= T or t not in sampled_rows[h]:
                            continue

                        a_t = sampled_rows[h][t].to(X_vo.device, dtype=X_vo.dtype).unsqueeze(0)  # [1, T]
                        S_th = (a_t @ X_vo).squeeze(0)  # [d_model]

                        O_h = WO[:, h*hD:(h+1)*hD] if WO is not None else None
                        if O_h is not None:
                            if torch.isnan(S_th).any():
                                continue

                            for c in C_out_vo:
                                r_h = (O_h.T @ c)
                                if torch.isnan(r_h).any():
                                    continue

                                sc = w_v / math.sqrt(hD)
                                constraints["vo"][h]["Xi_v"].append(S_th.detach().cpu())
                                constraints["vo"][h]["rv"].append(r_h.detach().cpu())
                                constraints["vo"][h]["sc_v"].append(torch.tensor([sc], dtype=torch.float32))

                # O constraint
                if merge_o and hasattr(samp, 'o_t') and samp.o_t:
                    for t in samp.o_t:
                        if t >= T or t not in sampled_rows[h]:
                            continue

                        a_t = sampled_rows[h][t].to(V_h.device, dtype=V_h.dtype).unsqueeze(0)  # [1, T]
                        u_th = (a_t @ V_h).squeeze(0)  # [hD]

                        if torch.isnan(u_th).any():
                            continue

                        for c in C_out_vo:
                            sc = w_o / math.sqrt(hD)
                            constraints["vo"][h]["c_vec"].append(c.detach().cpu())
                            constraints["vo"][h]["z_h"].append(u_th.detach().cpu())
                            constraints["vo"][h]["sc_o"].append(torch.tensor([sc], dtype=torch.float32))

        # =========================
        # FFN constraints
        # =========================
        if merge_f and X_ffn is not None:
            X_ffn_dev = X_ffn.to(ffn_device)

            gate_full = X_ffn_dev @ Wg.T if Wg is not None else None
            up_full = X_ffn_dev @ Wu.T if Wu is not None else None

            if hasattr(samp, 'ffn_t') and samp.ffn_t:
                for t in samp.ffn_t:
                    if t >= T:
                        continue

                    # down-proj related constraint
                    if gate_full is not None and up_full is not None:
                        g_t = gate_full[t]
                        u_t = up_full[t]
                        H_t = (torch.nn.functional.silu(g_t) * u_t)

                        if not torch.isnan(H_t).any():
                            for c in C_out_ffn_down:
                                constraints["ffn"]["H"].append(H_t.detach().cpu())
                                constraints["ffn"]["c"].append(c.detach().cpu())
                                constraints["ffn"]["sc"].append(torch.tensor([w_ffn / math.sqrt(d_ff)], dtype=torch.float32))

                    # gate-proj related constraint
                    if X_ffn_dev is not None:
                        x_t = X_ffn_dev[t]
                        if not torch.isnan(x_t).any():
                            for c in C_out_ffn_gate_up:
                                constraints["ffn"]["X_gate"].append(x_t.detach().cpu())
                                constraints["ffn"]["c_gate"].append(c.detach().cpu())
                                constraints["ffn"]["sc_gate"].append(torch.tensor([w_ffn / math.sqrt(d_model)], dtype=torch.float32))

                    # up-proj related constraint
                    if X_ffn_dev is not None:
                        x_t = X_ffn_dev[t]
                        if not torch.isnan(x_t).any():
                            for c in C_out_ffn_gate_up:
                                constraints["ffn"]["X_up"].append(x_t.detach().cpu())
                                constraints["ffn"]["c_up"].append(c.detach().cpu())
                                constraints["ffn"]["sc_up"].append(torch.tensor([w_ffn / math.sqrt(d_model)], dtype=torch.float32))

        # release per-sample temp tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if skipped_samples > 0:
        print(f"  ⚠️ Skipped {skipped_samples} over-length samples for layer {layer}")

    # =========================
    # Final stack / cat
    # =========================
    if merge_q or merge_k:
        for h in selected_heads:
            cons_h = constraints["qk"][h]

            for key in ["Xi_q", "kj", "Xj_k", "qi"]:
                if cons_h[key]:
                    cons_h[key] = torch.cat(cons_h[key], dim=0).contiguous()
                else:
                    cons_h[key] = torch.empty(0, dtype=torch.float32)

            for key in ["sc_q", "sc_k"]:
                if cons_h[key]:
                    cons_h[key] = torch.cat(cons_h[key], dim=0).contiguous()
                else:
                    cons_h[key] = torch.empty(0, 1, dtype=torch.float32)

    if merge_v or merge_o:
        for h in selected_heads:
            cons_h = constraints["vo"][h]

            for key in ["Xi_v", "rv", "c_vec", "z_h"]:
                if cons_h[key]:
                    cons_h[key] = torch.stack(cons_h[key], dim=0).contiguous()
                else:
                    cons_h[key] = torch.empty(0, dtype=torch.float32)

            for key in ["sc_v", "sc_o"]:
                if cons_h[key]:
                    cons_h[key] = torch.cat(cons_h[key], dim=0).contiguous()
                else:
                    cons_h[key] = torch.empty(0, 1, dtype=torch.float32)

    if merge_f:
        ffn_cons = constraints["ffn"]

        for key in ["H", "c", "X_gate", "c_gate", "X_up", "c_up"]:
            if ffn_cons[key]:
                ffn_cons[key] = torch.stack(ffn_cons[key], dim=0).contiguous()
            else:
                ffn_cons[key] = torch.empty(0, dtype=torch.float32)

        for key in ["sc", "sc_gate", "sc_up"]:
            if ffn_cons[key]:
                ffn_cons[key] = torch.cat(ffn_cons[key], dim=0).contiguous()
            else:
                ffn_cons[key] = torch.empty(0, 1, dtype=torch.float32)

    return constraints
