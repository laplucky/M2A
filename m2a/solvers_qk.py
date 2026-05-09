from typing import Dict, Tuple, Any

import torch


# ========== Vectorized A/AT operations (Q/K - original) ==========

@torch.no_grad()
def A_times_delta_qk_batched(delta_dQ: torch.Tensor, delta_dK: torch.Tensor,
                            cons_h: Dict[str, torch.Tensor], device: str = "cpu",
                            compute_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Vectorized A·Δ (use GEMM instead of scalar loops)"""
    y_list = []

    # Q constraint: y_q = scale_q * diag(Xi @ dQ @ kj^T)
    if cons_h["Xi_q"].numel() > 0:
        Xi = cons_h["Xi_q"].to(device, compute_dtype)     # [m_q, d_model]
        kj = cons_h["kj"].to(device, compute_dtype)       # [m_q, hD]
        sc = cons_h["sc_q"].to(device, compute_dtype).squeeze(-1)  # [m_q]

        # Matrix multiply: Xi @ dQ -> [m_q, hD]
        M = Xi @ delta_dQ.to(device, compute_dtype)       # [m_q, hD]
        yq = sc * (M * kj).sum(dim=1)                     # [m_q]
        y_list.append(yq)

    # K constraint: y_k = scale_k * diag(Xj @ dK @ qi^T)
    if cons_h["Xj_k"].numel() > 0:
        Xj = cons_h["Xj_k"].to(device, compute_dtype)     # [m_k, d_model]
        qi = cons_h["qi"].to(device, compute_dtype)       # [m_k, hD]
        sc = cons_h["sc_k"].to(device, compute_dtype).squeeze(-1)  # [m_k]

        M = Xj @ delta_dK.to(device, compute_dtype)       # [m_k, hD]
        yk = sc * (M * qi).sum(dim=1)                     # [m_k]
        y_list.append(yk)

    return torch.cat(y_list, dim=0) if y_list else torch.zeros(0, device=device, dtype=compute_dtype)

@torch.no_grad()
def AT_times_y_qk_batched(y: torch.Tensor, cons_h: Dict[str, torch.Tensor],
                         shapes: Tuple[int, int], device: str = "cpu",
                         compute_dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized A^T·y"""
    d_model, hD = shapes
    dQ = torch.zeros((d_model, hD), device=device, dtype=compute_dtype)
    dK = torch.zeros((d_model, hD), device=device, dtype=compute_dtype)
    idx = 0

    # Transpose for Q constraints: dQ += Xi^T @ diag(w * sc_q) @ kj
    if cons_h["Xi_q"].numel() > 0:
        m_q = cons_h["Xi_q"].shape[0]
        w = (y[idx:idx+m_q] * cons_h["sc_q"].squeeze(-1).to(device)).unsqueeze(1)  # [m_q, 1]
        Xi = cons_h["Xi_q"].to(device, compute_dtype)                              # [m_q, d_model]
        kj = cons_h["kj"].to(device, compute_dtype)                                # [m_q, hD]

        # Compute Xi^T @ (w * kj) via GEMM
        dQ += Xi.T @ (w * kj)                                                      # [d_model, hD]
        idx += m_q

    # Transpose for K constraints
    if cons_h["Xj_k"].numel() > 0:
        m_k = cons_h["Xj_k"].shape[0]
        w = (y[idx:idx+m_k] * cons_h["sc_k"].squeeze(-1).to(device)).unsqueeze(1)  # [m_k, 1]
        Xj = cons_h["Xj_k"].to(device, compute_dtype)                              # [m_k, d_model]
        qi = cons_h["qi"].to(device, compute_dtype)                                # [m_k, hD]

        dK += Xj.T @ (w * qi)                                                      # [d_model, hD]
        idx += m_k

    return dQ, dK


# ===== Q/K Dense/Cholesky explicit solvers (same pattern as FFN) =====

@torch.no_grad()
def q_dense_project(cons_h, task_dQ, lam=1e-4, device="cpu", compute_dtype=torch.float32):
    """
    Dense projection for Q constraints:
    cons_h["Xi_q"]: [m, d_model], cons_h["kj"]: [m, hD], cons_h["sc_q"]: [m]
    Δ = task_dQ ∈ R[d_model, hD]
    Gram: G = (s s^T) ⊙ (X X^T) ⊙ (KJ KJ^T) + λI
    rhs_i = s_i * < (X_i Δ), kj_i >
    A^T z = X^T @ ( (z ⊙ s)[:,None] ⊙ kj )
    """
    if cons_h["Xi_q"].numel() == 0:
        return task_dQ, {"residual_norm": 0.0, "solver": "dense_skip", "m": 0, "iterations": 0}

    X  = cons_h["Xi_q"].to(device, compute_dtype)          # [m, d_model]
    KJ = cons_h["kj"].to(device, compute_dtype)            # [m, hD]
    s  = cons_h["sc_q"].to(device, compute_dtype).squeeze(-1)  # [m]
    m  = X.size(0)

    XX = X @ X.T                                           # [m, m]
    KK = KJ @ KJ.T                                         # [m, m]
    G  = (XX * KK) * (s[:, None] * s[None, :]) + lam * torch.eye(m, device=device, dtype=compute_dtype)

    Δ  = task_dQ.to(device, compute_dtype)                 # [d_model, hD]
    M  = X @ Δ                                             # [m, hD]
    rhs = s * (M * KJ).sum(dim=1)                          # [m]

    try:
        L = torch.linalg.cholesky(G)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    except RuntimeError:
        z = torch.linalg.solve(G, rhs)

    w = z * s                                              # [m]
    dQ_w   = X.T @ (w[:, None] * KJ)                       # [d_model, hD]
    dQ_proj= Δ - dQ_w

    # Residual ||A dQ_proj||
    M2   = X @ dQ_proj                                     # [m, hD]
    resid= (s * (M2 * KJ).sum(dim=1)).norm().item()
    return dQ_proj.to(task_dQ.dtype), {"residual_norm": resid, "solver": "dense_cholesky", "m": m, "iterations": 1}


@torch.no_grad()
def k_dense_project(cons_h, task_dK, lam=1e-4, device="cpu", compute_dtype=torch.float32):
    """
    Dense projection for K constraints:
    cons_h["Xj_k"]: [m, d_model], cons_h["qi"]: [m, hD], cons_h["sc_k"]: [m]
    Δ = task_dK ∈ R[d_model, hD]
    Gram: G = (s s^T) ⊙ (X X^T) ⊙ (QI QI^T) + λI
    rhs_i = s_i * < (X_i Δ), qi_i >
    A^T z = X^T @ ( (z ⊙ s)[:,None] ⊙ qi )
    """
    if cons_h["Xj_k"].numel() == 0:
        return task_dK, {"residual_norm": 0.0, "solver": "dense_skip", "m": 0, "iterations": 0}

    X  = cons_h["Xj_k"].to(device, compute_dtype)          # [m, d_model]
    QI = cons_h["qi"].to(device, compute_dtype)            # [m, hD]
    s  = cons_h["sc_k"].to(device, compute_dtype).squeeze(-1)  # [m]
    m  = X.size(0)

    XX = X @ X.T
    QQ = QI @ QI.T
    G  = (XX * QQ) * (s[:, None] * s[None, :]) + lam * torch.eye(m, device=device, dtype=compute_dtype)

    Δ  = task_dK.to(device, compute_dtype)
    M  = X @ Δ                                             # [m, hD]
    rhs = s * (M * QI).sum(dim=1)

    try:
        L = torch.linalg.cholesky(G)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    except RuntimeError:
        z = torch.linalg.solve(G, rhs)

    w = z * s
    dK_w   = X.T @ (w[:, None] * QI)
    dK_proj= Δ - dK_w

    M2   = X @ dK_proj
    resid= (s * (M2 * QI).sum(dim=1)).norm().item()
    return dK_proj.to(task_dK.dtype), {"residual_norm": resid, "solver": "dense_cholesky", "m": m, "iterations": 1}


# ========== Vectorized CG solver (Q/K - original) ==========

def cg_single_head_batched(cons_h: Dict[str, torch.Tensor],
                          task_dQ: torch.Tensor, task_dK: torch.Tensor,
                          lambda_ridge: float = 1e-4, maxit: int = 100,
                          tol: float = 1e-5, device: str = "cpu",
                          compute_dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Vectorized single-head CG"""
    d_model, hD = task_dQ.shape

    # Convert task vectors to compute_dtype for CG
    task_dQ_compute = task_dQ.to(device, compute_dtype)
    task_dK_compute = task_dK.to(device, compute_dtype)

    # Right-hand side
    rhs = A_times_delta_qk_batched(task_dQ_compute, task_dK_compute, cons_h, device, compute_dtype)

    if rhs.numel() == 0:
        return task_dQ, task_dK, {
            "rhs": rhs.cpu(),
            "z": torch.tensor([]),
            "residual_norm": 0.0,
            "iterations": 0
        }

    def Mv(z):
        """Matrix-vector multiply: (AA^T + λI)z"""
        dQ_temp, dK_temp = AT_times_y_qk_batched(z, cons_h, (d_model, hD), device, compute_dtype)
        Az = A_times_delta_qk_batched(dQ_temp, dK_temp, cons_h, device, compute_dtype)
        return Az + lambda_ridge * z

    # Standard CG
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rs = (r * r).sum()

    for it in range(maxit):
        Ap = Mv(p)
        alpha = rs / ((p * Ap).sum() + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()

        if torch.sqrt(rs_new) <= tol * torch.sqrt((rhs * rhs).sum() + 1e-12):
            break

        beta = rs_new / (rs + 1e-12)
        p = r + beta * p
        rs = rs_new

    # Projection
    dQ_w, dK_w = AT_times_y_qk_batched(x, cons_h, (d_model, hD), device, compute_dtype)
    dQ_proj_compute = task_dQ_compute - dQ_w
    dK_proj_compute = task_dK_compute - dK_w

    # Residual check
    residual = A_times_delta_qk_batched(dQ_proj_compute, dK_proj_compute, cons_h, device, compute_dtype)

    # Back to original dtype
    return dQ_proj_compute.to(task_dQ.dtype), dK_proj_compute.to(task_dK.dtype), {
        "rhs": rhs.cpu(),
        "z": x.cpu(),
        "residual_norm": residual.norm().item(),
        "iterations": it + 1
    }
