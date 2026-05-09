import os
import gc

import torch


# ========== Basic utilities ==========

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def cleanup_memory():
    """Clean up memory and GPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_memory_status(stage: str):
    """Print memory status"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"🔧 [{stage}] GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print(f"🔧 [{stage}] Using CPU mode")

def get_model_config_attr(config, attr_name, default=None):
    """
    Get config attribute with compatibility for Qwen3.5 nested text_config structure.

    Args:
        config: Model config object
        attr_name: Attribute name to retrieve
        default: Default value if attribute not found

    Returns:
        Attribute value
    """
    # First try text_config for Qwen3.5 compatibility
    if hasattr(config, 'text_config') and config.text_config is not None:
        if hasattr(config.text_config, attr_name):
            return getattr(config.text_config, attr_name)

    # Fall back to direct config access for standard models
    if hasattr(config, attr_name):
        return getattr(config, attr_name)

    # Return default if not found
    if default is not None:
        return default

    raise AttributeError(f"Config object has no attribute '{attr_name}'")

def get_layer_modules(model, layer_idx: int):
    """
    Get attention and MLP modules from a layer with compatibility for different model architectures.

    Args:
        model: The model object
        layer_idx: Layer index

    Returns:
        Tuple of (attention_module, mlp_module)
        Returns (None, mlp_module) if the layer doesn't have compatible attention
    """
    layer = model.model.layers[layer_idx]

    # Try different attention attribute names
    attn = None
    attn_attr_name = None
    for attr_name in ['self_attn', 'attn', 'attention', 'self_attention', 'linear_attn']:
        if hasattr(layer, attr_name):
            candidate = getattr(layer, attr_name)
            # Check if this attention module has standard q/k/v/o projections
            # Skip if it doesn't (e.g., Qwen3.5 linear_attn uses special projections)
            if all(hasattr(candidate, proj) for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                attn = candidate
                attn_attr_name = attr_name
                break

    # For Qwen3.5: linear_attn layers don't have standard projections, return None
    if attn is None:
        # Check if this is a linear_attn layer that we need to skip
        if hasattr(layer, 'linear_attn'):
            # Return None for attention but still return MLP
            mlp = None
            for attr_name in ['mlp', 'feed_forward', 'ffn']:
                if hasattr(layer, attr_name):
                    mlp = getattr(layer, attr_name)
                    break
            return None, mlp

    if attn is None:
        raise AttributeError(
            f"Could not find compatible attention module in layer {layer_idx}. "
            f"Tried: self_attn, attn, attention, self_attention, linear_attn. "
            f"Note: For Qwen3.5, only full attention layers (not linear_attn) are supported."
        )

    # Try to get MLP module
    mlp = None
    for attr_name in ['mlp', 'feed_forward', 'ffn']:
        if hasattr(layer, attr_name):
            mlp = getattr(layer, attr_name)
            break

    if mlp is None:
        raise AttributeError(
            f"Could not find MLP module in layer {layer_idx}. "
            f"Tried: mlp, feed_forward, ffn"
        )

    return attn, mlp


def set_strict_runtime():
    """Set strict runtime environment to ensure numerical consistency"""
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
    torch.use_deterministic_algorithms(False)
