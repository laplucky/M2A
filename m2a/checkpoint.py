import os
import time
from typing import List, Dict, Any, Optional

import torch


def save_checkpoint(
    checkpoint_dir: str,
    phase: str,
    layer: int,
    processed_layers: List[int],
    data: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """Save checkpoint data to disk"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{phase}.pt")

    checkpoint = {
        "data": data,
        "config": config,
        "processed_layers": processed_layers,
        "timestamp": time.time(),
        "phase": phase,
        "layer": layer
    }

    torch.save(checkpoint, checkpoint_file)
    print(f"💾 Checkpoint saved: {checkpoint_file} (phase={phase}, layer={layer}, processed={len(processed_layers)})")


def load_checkpoint(checkpoint_dir: str, phase: str, expected_config: Dict[str, Any], selected_layers: List[int]) -> Optional[Dict[str, Any]]:
    """Load checkpoint from disk, returns None if not found or config mismatch"""
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{phase}.pt")

    if not os.path.exists(checkpoint_file):
        return None

    try:
        checkpoint = torch.load(checkpoint_file, map_location="cpu")

        # Validate config match
        checkpoint_config = checkpoint.get("config", {})
        config_keys = ["beta", "k_threshold", "window_size", "merge_types", "selected_layers", "selected_heads"]
        for key in config_keys:
            if key in expected_config and key in checkpoint_config:
                if expected_config[key] != checkpoint_config[key]:
                    print(f"⚠️  Config mismatch for {key}: expected {expected_config[key]}, got {checkpoint_config[key]}")
                    print(f"🗑️  Ignoring checkpoint due to config mismatch")
                    return None

        # Validate that all selected layers have been processed
        processed_layers = checkpoint.get("processed_layers", [])
        checkpoint_layer = checkpoint.get("layer", -1)

        print(f"✅ Checkpoint loaded: {checkpoint_file}")
        print(f"   Phase: {checkpoint['phase']}, Layer: {checkpoint['layer']}, Processed: {processed_layers}")
        return checkpoint
    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
        return None


def cleanup_checkpoint(checkpoint_dir: str, phase: str) -> None:
    """Remove checkpoint for a specific phase"""
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{phase}.pt")
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"🗑️  Checkpoint removed: {checkpoint_file}")
