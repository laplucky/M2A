import math
from typing import List, Dict, Any

import torch


def compute_frobenius_norm_layer(layer_task_vectors: Dict[str, Any]) -> float:
    """Compute Frobenius norm for all task vectors in a layer"""
    total_norm_sq = 0.0

    # QK task vectors
    if "qk" in layer_task_vectors:
        for head_data in layer_task_vectors["qk"].values():
            for key in ["dQ", "dK"]:
                if key in head_data and head_data[key] is not None:
                    total_norm_sq += torch.sum(head_data[key] ** 2).item()

    # VO task vectors
    if "vo" in layer_task_vectors:
        for head_data in layer_task_vectors["vo"].values():
            for key in ["dV", "dO"]:
                if key in head_data and head_data[key] is not None:
                    total_norm_sq += torch.sum(head_data[key] ** 2).item()

    # FFN task vectors
    if "ffn" in layer_task_vectors:
        for key in ["dGate", "dUp", "dDown_T"]:
            if key in layer_task_vectors["ffn"] and layer_task_vectors["ffn"][key] is not None:
                total_norm_sq += torch.sum(layer_task_vectors["ffn"][key] ** 2).item()

    return math.sqrt(total_norm_sq)


def compute_cosine_similarity_layer(agent_task_vectors: Dict[str, Any],
                                    reason_task_vectors: Dict[str, Any]) -> float:
    """Compute cosine similarity between agent and reasoning task vectors for a layer"""
    dot_product = 0.0
    norm_agent_sq = 0.0
    norm_reason_sq = 0.0

    # QK similarity
    if "qk" in agent_task_vectors and "qk" in reason_task_vectors:
        for h in agent_task_vectors["qk"].keys():
            if h in reason_task_vectors["qk"]:
                agent_h = agent_task_vectors["qk"][h]
                reason_h = reason_task_vectors["qk"][h]

                for key in ["dQ", "dK"]:
                    if key in agent_h and key in reason_h:
                        agent_vec = agent_h[key]
                        reason_vec = reason_h[key]

                        dot_product += torch.sum(agent_vec * reason_vec).item()
                        norm_agent_sq += torch.sum(agent_vec ** 2).item()
                        norm_reason_sq += torch.sum(reason_vec ** 2).item()

    # VO similarity
    if "vo" in agent_task_vectors and "vo" in reason_task_vectors:
        for h in agent_task_vectors["vo"].keys():
            if h in reason_task_vectors["vo"]:
                agent_h = agent_task_vectors["vo"][h]
                reason_h = reason_task_vectors["vo"][h]

                for key in ["dV", "dO"]:
                    if key in agent_h and key in reason_h:
                        agent_vec = agent_h[key]
                        reason_vec = reason_h[key]

                        dot_product += torch.sum(agent_vec * reason_vec).item()
                        norm_agent_sq += torch.sum(agent_vec ** 2).item()
                        norm_reason_sq += torch.sum(reason_vec ** 2).item()

    # FFN similarity
    if "ffn" in agent_task_vectors and "ffn" in reason_task_vectors:
        agent_ffn = agent_task_vectors["ffn"]
        reason_ffn = reason_task_vectors["ffn"]

        for key in ["dGate", "dUp", "dDown_T"]:
            if key in agent_ffn and key in reason_ffn:
                agent_vec = agent_ffn[key]
                reason_vec = reason_ffn[key]

                if agent_vec is not None and reason_vec is not None:
                    dot_product += torch.sum(agent_vec * reason_vec).item()
                    norm_agent_sq += torch.sum(agent_vec ** 2).item()
                    norm_reason_sq += torch.sum(reason_vec ** 2).item()

    # Compute cosine similarity
    norm_agent = math.sqrt(norm_agent_sq) if norm_agent_sq > 0 else 1e-10
    norm_reason = math.sqrt(norm_reason_sq) if norm_reason_sq > 0 else 1e-10

    cosine_sim = dot_product / (norm_agent * norm_reason)
    return cosine_sim


def smooth_similarity_scores(similarities: List[float], window_size: int = 3) -> List[float]:
    """Apply moving average smoothing to similarity scores"""
    if len(similarities) < window_size:
        return similarities

    smoothed = []
    for i in range(len(similarities)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(similarities), i + window_size // 2 + 1)
        window_values = similarities[start_idx:end_idx]
        smoothed.append(sum(window_values) / len(window_values))

    return smoothed


def compute_layer_selection_mask(similarities: List[float],
                                 k: float = 0.5,
                                 window_size: int = 3) -> List[int]:
    """
    Compute binary mask for layer selection based on similarity scores.
    Filters out layers with cliff-like drops in similarity (format conflict layers).

    Args:
        similarities: List of similarity scores for each layer
        k: Threshold parameter (default 0.5)
        window_size: Window size for moving average smoothing (default 3)

    Returns:
        Binary mask (0 or 1) for each layer
    """
    # Step 1: Smooth the similarity scores
    smoothed = smooth_similarity_scores(similarities, window_size)

    # Step 2: Compute global statistics
    mean_sim = sum(smoothed) / len(smoothed)
    variance = sum((s - mean_sim) ** 2 for s in smoothed) / len(smoothed)
    std_sim = math.sqrt(variance)

    # Step 3: Generate binary mask
    threshold = mean_sim - k * std_sim
    mask = [1 if s >= threshold else 0 for s in smoothed]

    return mask, smoothed, mean_sim, std_sim, threshold


def compute_dynamic_alpha(norm_agent: float, norm_reason: float, beta: float = 1.0) -> float:
    """
    Compute dynamic merge coefficient based on Frobenius norm ratio.
    α_l = β · ||ΔW_agent||_F / ||ΔW_reason||_F
    """
    if norm_reason < 1e-10:
        return 0.0

    alpha = beta * (norm_agent / norm_reason)
    return alpha
