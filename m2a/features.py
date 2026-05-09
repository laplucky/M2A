from typing import List

import torch

from .utils import set_strict_runtime


def collect_layer_features_with_hooks(
    model,
    input_ids: torch.Tensor,
    selected_layers: List[int],
    merge_types: str = "qkvof",
    max_seq_len: int = 7168
):
    """Collect layer internal features using hooks.

    Key changes:
      1. Do NOT build full [H, T, T] attention weights.
      2. For VO constraints, only cache q_proj_out / k_proj_out.
      3. Immediately move large tensors to CPU when possible.
      4. Explicitly disable KV cache during forward.
    """
    seq_length = input_ids.shape[-1]
    if seq_length > max_seq_len:
        print(f"⚠️  Sequence length {seq_length} exceeds limit {max_seq_len}; skipping")
        return {layer_idx: {} for layer_idx in selected_layers}

    set_strict_runtime()
    features = {}
    hooks = []

    need_qk = 'q' in merge_types.lower() or 'k' in merge_types.lower()
    need_vo = 'v' in merge_types.lower() or 'o' in merge_types.lower()
    need_ffn = 'f' in merge_types.lower()

    def register_strict_layer_hooks(layer_idx, layer):
        feat_bucket = features.setdefault(layer_idx, {})
        layer_hooks = []

        # 1) pre-LN output for QK / VO
        if need_qk or need_vo:
            def hook_attn_input_ln(module, inp, out):
                # out shape usually [B, T, d_model]
                feat_bucket["attn_input"] = out[0].detach().cpu()  # [T, d_model]
            h1 = layer.input_layernorm.register_forward_hook(hook_attn_input_ln)
            layer_hooks.append(h1)

        # 2) post-attention LN output for FFN
        if need_ffn:
            def hook_ffn_input_ln(module, inp, out):
                feat_bucket["ffn_input"] = out[0].detach().cpu()  # [T, d_model]
            h2 = layer.post_attention_layernorm.register_forward_hook(hook_ffn_input_ln)
            layer_hooks.append(h2)

            def hook_gate_output(module, inp, out):
                feat_bucket["gate_output"] = out.detach().cpu()

            def hook_up_output(module, inp, out):
                feat_bucket["up_output"] = out.detach().cpu()

            layer_mlp = layer.mlp
            h_gate = layer_mlp.gate_proj.register_forward_hook(hook_gate_output)
            h_up = layer_mlp.up_proj.register_forward_hook(hook_up_output)
            layer_hooks.extend([h_gate, h_up])

        # 3) For VO constraints, keep raw Q/K projection outputs only
        if need_vo:
            def hook_q_proj(module, inp, out):
                feat_bucket["q_proj_out"] = out.detach()

            def hook_k_proj(module, inp, out):
                feat_bucket["k_proj_out"] = out.detach()

            # Get attention module with compatibility
            layer_attn = None
            for attr_name in ['self_attn', 'attn', 'attention', 'self_attention']:
                if hasattr(layer, attr_name):
                    layer_attn = getattr(layer, attr_name)
                    break

            if layer_attn is None:
                raise AttributeError(f"Could not find attention module in layer")

            h3 = layer_attn.q_proj.register_forward_hook(hook_q_proj)
            h4 = layer_attn.k_proj.register_forward_hook(hook_k_proj)
            layer_hooks.extend([h3, h4])

        return layer_hooks

    for layer_idx in selected_layers:
        layer_obj = model.model.layers[layer_idx]
        hooks.extend(register_strict_layer_hooks(layer_idx, layer_obj))

    try:
        with torch.no_grad():
            print(input_ids.shape)
            _ = model.model(input_ids=input_ids, use_cache=False)

            # Immediately offload q/k to CPU after forward
            for layer_idx in selected_layers:
                if layer_idx not in features:
                    continue
                if "q_proj_out" in features[layer_idx]:
                    features[layer_idx]["q_proj_out"] = features[layer_idx]["q_proj_out"].cpu()
                if "k_proj_out" in features[layer_idx]:
                    features[layer_idx]["k_proj_out"] = features[layer_idx]["k_proj_out"].cpu()

    finally:
        for hook in hooks:
            hook.remove()

    return features



def compute_sampled_attention_rows_from_qk(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    config,
    selected_heads: List[int],
    sampled_ts: List[int],
    device: str,
):
    """
    Compute exact attention rows only for sampled token positions.

    Args:
        q_out: [B, T, H*head_dim] or [T, H*head_dim]
        k_out: [B, T, H_kv*head_dim] or [T, H_kv*head_dim]
        config: model config
        selected_heads: heads to compute
        sampled_ts: token positions t to compute attention rows for
        device: compute device

    Returns:
        rows_by_head: Dict[int, Dict[int, Tensor]]
            head -> t -> attention row of shape [T]
    """
    if q_out.dim() == 3:
        q_out = q_out[0]   # [T, H*head_dim]
    if k_out.dim() == 3:
        k_out = k_out[0]   # [T, H_kv*head_dim]

    T = q_out.shape[0]
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)

    q = q_out.view(T, num_heads, head_dim).permute(1, 0, 2).contiguous()      # [H, T, d]
    k = k_out.view(T, num_kv_heads, head_dim).permute(1, 0, 2).contiguous()   # [H_kv, T, d]

    # GQA expand
    if num_kv_heads < num_heads:
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=0)  # [H, T, d]

    # compute dtype
    original_dtype = q.dtype
    compute_dtype = torch.bfloat16 if original_dtype in [torch.float16, torch.bfloat16] else torch.float32

    q = q.to(device=device, dtype=compute_dtype)
    k = k.to(device=device, dtype=compute_dtype)

    valid_ts = sorted({int(t) for t in sampled_ts if 0 <= int(t) < T})
    rows_by_head = {h: {} for h in selected_heads}
    scale = head_dim ** 0.5

    for h in selected_heads:
        q_h = q[h]  # [T, d]
        k_h = k[h]  # [T, d]

        for t in valid_ts:
            q_t = q_h[t:t+1]  # [1, d]
            scores = (q_t @ k_h.transpose(0, 1)).squeeze(0) / scale  # [T]

            # causal mask: only attend to positions <= t
            if t + 1 < T:
                scores[t+1:] = torch.finfo(scores.dtype).min

            # stable softmax on valid prefix
            prefix = scores[:t+1]
            row_max = prefix.max()
            probs_prefix = torch.exp(prefix - row_max)
            denom = probs_prefix.sum()

            if denom.item() == 0:
                attn_row = torch.zeros(T, device=device, dtype=compute_dtype)
                attn_row[t] = 1.0
            else:
                attn_row = torch.zeros(T, device=device, dtype=compute_dtype)
                attn_row[:t+1] = probs_prefix / denom

            rows_by_head[h][t] = attn_row.cpu()

    return rows_by_head
