import torch


# ========== Vectorized A/AT operations (V/O - new) ==========

# ===== V: ΔW_V^{h'} ∈ R[d_model, hD]
@torch.no_grad()
def A_times_delta_v(delta_dV, cons_h, device="cpu", compute_dtype=torch.float32):
    y = []
    if cons_h["Xi_v"].numel():
        Xi = cons_h["Xi_v"].to(device, compute_dtype)      # [m, d_model]
        rv = cons_h["rv"].to(device, compute_dtype)        # [m, hD]
        sc = cons_h["sc_v"].to(device, compute_dtype).squeeze(-1)  # [m]
        M  = Xi @ delta_dV.to(device, compute_dtype)       # [m, hD]
        yv = sc * (M * rv).sum(dim=1)                      # [m]
        y.append(yv)
    return torch.cat(y, dim=0) if y else torch.zeros(0, device=device, dtype=compute_dtype)

@torch.no_grad()
def AT_times_y_v(y, cons_h, d_model, hD, device="cpu", compute_dtype=torch.float32):
    dV = torch.zeros((d_model, hD), device=device, dtype=compute_dtype)
    idx = 0
    if cons_h["Xi_v"].numel():
        m = cons_h["Xi_v"].shape[0]
        w = (y[idx:idx+m] * cons_h["sc_v"].squeeze(-1).to(device)).unsqueeze(1)
        Xi = cons_h["Xi_v"].to(device, compute_dtype)    # [m,d_model]
        rv = cons_h["rv"].to(device, compute_dtype)      # [m,hD]
        dV += Xi.T @ (w * rv)             # [d_model,hD]
        idx += m
    return dV

def cg_v(cons_h, task_dV, lam=1e-4, maxit=100, tol=1e-5, device="cpu", compute_dtype=torch.float32):
    # Convert task_dV to compute_dtype for CG
    task_dV_compute = task_dV.to(device, compute_dtype)
    rhs = A_times_delta_v(task_dV_compute, cons_h, device, compute_dtype)
    if rhs.numel()==0:
        return task_dV, {"rhs":rhs.cpu(), "z":torch.tensor([]), "residual_norm":0.0, "iterations":0}
    def Mv(z):
        dV = AT_times_y_v(z, cons_h, task_dV_compute.size(0), task_dV_compute.size(1), device, compute_dtype)
        Az = A_times_delta_v(dV, cons_h, device, compute_dtype)
        return Az + lam * z
    # CG
    x = torch.zeros_like(rhs); r=rhs.clone(); p=r.clone(); rs=(r*r).sum()
    it=0
    for it in range(maxit):
        Ap = Mv(p); alpha = rs / ((p*Ap).sum()+1e-12)
        x = x + alpha*p; r = r - alpha*Ap
        rs_new = (r*r).sum()
        if torch.sqrt(rs_new) <= tol * torch.sqrt((rhs*rhs).sum()+1e-12): break
        p = r + (rs_new/(rs+1e-12))*p; rs = rs_new
    dV_w = AT_times_y_v(x, cons_h, task_dV_compute.size(0), task_dV_compute.size(1), device, compute_dtype)
    dV_proj = task_dV_compute - dV_w
    res = A_times_delta_v(dV_proj, cons_h, device, compute_dtype)
    # Back to original dtype
    return dV_proj.to(task_dV.dtype), {"rhs":rhs.cpu(),"z":x.cpu(),"residual_norm":res.norm().item(),"iterations":it+1}

# ===== O: ΔW_{O,h} ∈ R[d_model, hD] (by column block)
@torch.no_grad()
def A_times_delta_o(delta_dO, cons_h, device="cpu", compute_dtype=torch.float32):
    y = []
    if cons_h["c_vec"].numel():
        C  = cons_h["c_vec"].to(device, compute_dtype)   # [m,d_model]
        zh = cons_h["z_h"].to(device, compute_dtype)     # [m,hD]
        sc = cons_h["sc_o"].to(device, compute_dtype).squeeze(-1)
        M  = C @ delta_dO.to(device, compute_dtype)      # [m,hD]
        yo = sc * (M * zh).sum(dim=1)
        y.append(yo)
    return torch.cat(y, dim=0) if y else torch.zeros(0, device=device, dtype=compute_dtype)

@torch.no_grad()
def AT_times_y_o(y, cons_h, d_model, hD, device="cpu", compute_dtype=torch.float32):
    dO = torch.zeros((d_model, hD), device=device, dtype=compute_dtype)
    idx = 0
    if cons_h["c_vec"].numel():
        m = cons_h["c_vec"].shape[0]
        w = (y[idx:idx+m] * cons_h["sc_o"].squeeze(-1).to(device)).unsqueeze(1)
        C = cons_h["c_vec"].to(device, compute_dtype)
        zh= cons_h["z_h"].to(device, compute_dtype)
        dO += C.T @ (w * zh)
        idx += m
    return dO

def cg_o(cons_h, task_dO, lam=1e-4, maxit=100, tol=1e-5, device="cpu", compute_dtype=torch.float32):
    # Convert task_dO to compute_dtype for CG
    task_dO_compute = task_dO.to(device, compute_dtype)
    rhs = A_times_delta_o(task_dO_compute, cons_h, device, compute_dtype)
    if rhs.numel()==0:
        return task_dO, {"rhs":rhs.cpu(),"z":torch.tensor([]),"residual_norm":0.0,"iterations":0}
    def Mv(z):
        dO = AT_times_y_o(z, cons_h, task_dO_compute.size(0), task_dO_compute.size(1), device, compute_dtype)
        Az = A_times_delta_o(dO, cons_h, device, compute_dtype)
        return Az + lam * z
    # CG
    x = torch.zeros_like(rhs); r=rhs.clone(); p=r.clone(); rs=(r*r).sum()
    it=0
    for it in range(maxit):
        Ap = Mv(p); alpha = rs / ((p*Ap).sum()+1e-12)
        x = x + alpha*p; r = r - alpha*Ap
        rs_new = (r*r).sum()
        if torch.sqrt(rs_new) <= tol * torch.sqrt((rhs*rhs).sum()+1e-12): break
        p = r + (rs_new/(rs+1e-12))*p; rs = rs_new
    dO_w = AT_times_y_o(x, cons_h, task_dO_compute.size(0), task_dO_compute.size(1), device, compute_dtype)
    dO_proj = task_dO_compute - dO_w
    res = A_times_delta_o(dO_proj, cons_h, device, compute_dtype)
    # Back to original dtype
    return dO_proj.to(task_dO.dtype), {"rhs":rhs.cpu(),"z":x.cpu(),"residual_norm":res.norm().item(),"iterations":it+1}


# ===== V/O Dense/Cholesky explicit solvers (same pattern as FFN) =====

@torch.no_grad()
def v_dense_project(cons_h, task_dV, lam=1e-4, device="cpu", compute_dtype=torch.float32):
    """
    Dense projection for V constraints:
    cons_h["Xi_v"]: [m, d_model], cons_h["rv"]: [m, hD], cons_h["sc_v"]: [m]
    Δ = task_dV ∈ R[d_model, hD]
    Gram: G = (s s^T) ⊙ (X X^T) ⊙ (RV RV^T) + λI
    rhs_i = s_i * < (X_i Δ), rv_i >
    A^T z = X^T @ ( (z ⊙ s)[:,None] ⊙ rv )
    """
    if cons_h["Xi_v"].numel() == 0:
        return task_dV, {"residual_norm": 0.0, "solver": "dense_skip", "m": 0, "iterations": 0}

    X  = cons_h["Xi_v"].to(device, compute_dtype)
    RV = cons_h["rv"].to(device, compute_dtype)
    s  = cons_h["sc_v"].to(device, compute_dtype).squeeze(-1)
    m  = X.size(0)

    XX = X @ X.T
    RR = RV @ RV.T
    G  = (XX * RR) * (s[:, None] * s[None, :]) + lam * torch.eye(m, device=device, dtype=compute_dtype)

    Δ  = task_dV.to(device, compute_dtype)
    M  = X @ Δ
    rhs = s * (M * RV).sum(dim=1)

    try:
        L = torch.linalg.cholesky(G)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    except RuntimeError:
        z = torch.linalg.solve(G, rhs)

    w = z * s
    dV_w   = X.T @ (w[:, None] * RV)
    dV_proj= Δ - dV_w

    M2   = X @ dV_proj
    resid= (s * (M2 * RV).sum(dim=1)).norm().item()
    return dV_proj.to(task_dV.dtype), {"residual_norm": resid, "solver": "dense_cholesky", "m": m, "iterations": 1}


@torch.no_grad()
def o_dense_project(cons_h, task_dO, lam=1e-4, device="cpu", compute_dtype=torch.float32):
    """
    Dense projection for O constraints:
    cons_h["c_vec"]: [m, d_model], cons_h["z_h"]: [m, hD], cons_h["sc_o"]: [m]
    Δ = task_dO ∈ R[d_model, hD]
    Gram: G = (s s^T) ⊙ (C C^T) ⊙ (Z Z^T) + λI
    rhs_i = s_i * < (C_i Δ), z_i >
    A^T z = C^T @ ( (z ⊙ s)[:,None] ⊙ Z )
    """
    if cons_h["c_vec"].numel() == 0:
        return task_dO, {"residual_norm": 0.0, "solver": "dense_skip", "m": 0, "iterations": 0}

    C  = cons_h["c_vec"].to(device, compute_dtype)
    Z  = cons_h["z_h"].to(device, compute_dtype)
    s  = cons_h["sc_o"].to(device, compute_dtype).squeeze(-1)
    m  = C.size(0)

    CC = C @ C.T
    ZZ = Z @ Z.T
    G  = (CC * ZZ) * (s[:, None] * s[None, :]) + lam * torch.eye(m, device=device, dtype=compute_dtype)

    Δ  = task_dO.to(device, compute_dtype)
    M  = C @ Δ
    rhs = s * (M * Z).sum(dim=1)

    try:
        L = torch.linalg.cholesky(G)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    except RuntimeError:
        z = torch.linalg.solve(G, rhs)

    w = z * s
    dO_w   = C.T @ (w[:, None] * Z)
    dO_proj= Δ - dO_w

    M2   = C @ dO_proj
    resid= (s * (M2 * Z).sum(dim=1)).norm().item()
    return dO_proj.to(task_dO.dtype), {"residual_norm": resid, "solver": "dense_cholesky", "m": m, "iterations": 1}
