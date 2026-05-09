import torch


# ===== FFN-Gate: ΔW_gate ∈ R[d_ff, d_model]
@torch.no_grad()
def A_times_delta_ffn_gate(delta_dGate, cons, device="cpu", compute_dtype=torch.float32):
    """A(dW_gate) = [c_i^T @ (dW_gate @ x_i)] for Gate constraints"""
    y = []
    if cons["X_gate"].numel() > 0:
        X = cons["X_gate"].to(device, compute_dtype)        # [m, d_model]
        C = cons["c_gate"].to(device, compute_dtype)        # [m, d_ff] (directions in FFN hidden space)
        sc = cons["sc_gate"].to(device, compute_dtype).squeeze(-1)

        # dW_gate @ X.T = [d_ff, d_model] @ [d_model, m] = [d_ff, m]
        M = delta_dGate.to(device, compute_dtype) @ X.T     # [d_ff, m]
        # For each i: c_i^T @ (dW_gate @ x_i) = C[i] @ M[:, i]
        yf = sc * (C * M.T).sum(dim=1)  # [m]
        y.append(yf)
    return torch.cat(y, dim=0) if y else torch.zeros(0, device=device, dtype=compute_dtype)

@torch.no_grad()
def AT_times_y_ffn_gate(y, cons, d_ff, d_model, device="cpu", compute_dtype=torch.float32):
    """Vectorized A^T·y for Gate constraints"""
    if cons["X_gate"].numel() == 0:
        return torch.zeros((d_ff, d_model), device=device, dtype=compute_dtype)

    # y, sc_gate: [m]; X_gate: [m,d_model]; c_gate: [m,d_ff]
    w = (y * cons["sc_gate"].squeeze(-1).to(device)).to(compute_dtype)        # [m]
    X = cons["X_gate"].to(device, compute_dtype)                               # [m, d_model]
    C = cons["c_gate"].to(device, compute_dtype)                               # [m, d_ff]
    # sum_i w[i] * (c_i ⊗ x_i^T) == C^T @ (diag(w) @ X) == C.T @ (w[:,None]*X)
    return C.T @ (w[:, None] * X)                                             # [d_ff, d_model]

def cg_ffn_gate(cons, task_dGate, lam=1e-4, maxit=100, tol=1e-5, device="cpu", compute_dtype=torch.float32):
    """CG solver for Gate"""
    task_dGate_compute = task_dGate.to(device, compute_dtype)
    rhs = A_times_delta_ffn_gate(task_dGate_compute, cons, device, compute_dtype)
    if rhs.numel() == 0:
        return task_dGate, {"rhs":rhs.cpu(),"z":torch.tensor([]),"residual_norm":0.0,"iterations":0}

    def Mv(z):
        dG = AT_times_y_ffn_gate(z, cons, task_dGate_compute.size(0), task_dGate_compute.size(1), device, compute_dtype)
        Az = A_times_delta_ffn_gate(dG, cons, device, compute_dtype)
        return Az + lam * z

    # CG
    x = torch.zeros_like(rhs); r = rhs.clone(); p = r.clone()
    rs = (r*r).sum()
    for it in range(maxit):
        Ap = Mv(p); alpha = rs / ((p*Ap).sum() + 1e-12)
        x += alpha * p; r -= alpha * Ap; rs_new = (r*r).sum()
        if torch.sqrt(rs_new) <= tol * torch.sqrt((rhs*rhs).sum()+1e-12): break
        p = r + (rs_new/(rs+1e-12))*p; rs = rs_new

    dG_w = AT_times_y_ffn_gate(x, cons, task_dGate_compute.size(0), task_dGate_compute.size(1), device, compute_dtype)
    dG_proj = task_dGate_compute - dG_w
    res = A_times_delta_ffn_gate(dG_proj, cons, device, compute_dtype)
    return dG_proj.to(task_dGate.dtype), {"rhs":rhs.cpu(),"z":x.cpu(),"residual_norm":res.norm().item(),"iterations":it+1}

# ===== FFN-Up: ΔW_up ∈ R[d_ff, d_model] (similar to Gate)
@torch.no_grad()
def A_times_delta_ffn_up(delta_dUp, cons, device="cpu", compute_dtype=torch.float32):
    """A(dW_up) for Up constraints"""
    y = []
    if cons["X_up"].numel() > 0:
        X = cons["X_up"].to(device, compute_dtype)        # [m, d_model]
        C = cons["c_up"].to(device, compute_dtype)        # [m, d_ff]
        sc = cons["sc_up"].to(device, compute_dtype).squeeze(-1)

        M = delta_dUp.to(device, compute_dtype) @ X.T     # [d_ff, m]
        yf = sc * (C * M.T).sum(dim=1)                    # [m]
        y.append(yf)
    return torch.cat(y, dim=0) if y else torch.zeros(0, device=device, dtype=compute_dtype)

@torch.no_grad()
def AT_times_y_ffn_up(y, cons, d_ff, d_model, device="cpu", compute_dtype=torch.float32):
    """Vectorized A^T·y for Up constraints"""
    if cons["X_up"].numel() == 0:
        return torch.zeros((d_ff, d_model), device=device, dtype=compute_dtype)

    w = (y * cons["sc_up"].squeeze(-1).to(device)).to(compute_dtype)          # [m]
    X = cons["X_up"].to(device, compute_dtype)                                 # [m, d_model]
    C = cons["c_up"].to(device, compute_dtype)                                 # [m, d_ff]
    return C.T @ (w[:, None] * X)                                             # [d_ff, d_model]

def cg_ffn_up(cons, task_dUp, lam=1e-4, maxit=100, tol=1e-5, device="cpu", compute_dtype=torch.float32):
    """CG solver for Up"""
    task_dUp_compute = task_dUp.to(device, compute_dtype)
    rhs = A_times_delta_ffn_up(task_dUp_compute, cons, device, compute_dtype)
    if rhs.numel() == 0:
        return task_dUp, {"rhs":rhs.cpu(),"z":torch.tensor([]),"residual_norm":0.0,"iterations":0}

    def Mv(z):
        dU = AT_times_y_ffn_up(z, cons, task_dUp_compute.size(0), task_dUp_compute.size(1), device, compute_dtype)
        Az = A_times_delta_ffn_up(dU, cons, device, compute_dtype)
        return Az + lam * z

    # CG
    x = torch.zeros_like(rhs); r = rhs.clone(); p = r.clone()
    rs = (r*r).sum()
    for it in range(maxit):
        Ap = Mv(p); alpha = rs / ((p*Ap).sum() + 1e-12)
        x += alpha * p; r -= alpha * Ap; rs_new = (r*r).sum()
        if torch.sqrt(rs_new) <= tol * torch.sqrt((rhs*rhs).sum()+1e-12): break
        p = r + (rs_new/(rs+1e-12))*p; rs = rs_new

    dU_w = AT_times_y_ffn_up(x, cons, task_dUp_compute.size(0), task_dUp_compute.size(1), device, compute_dtype)
    dU_proj = task_dUp_compute - dU_w
    res = A_times_delta_ffn_up(dU_proj, cons, device, compute_dtype)
    return dU_proj.to(task_dUp.dtype), {"rhs":rhs.cpu(),"z":x.cpu(),"residual_norm":res.norm().item(),"iterations":it+1}

# ===== FFN-Down: ΔW_down ∈ R[d_model, d_ff] (we use transposed ΔW_down^T for shape match)
@torch.no_grad()
def A_times_delta_ffn_down(delta_dDown_T, cons, device="cpu", compute_dtype=torch.float32):
    # delta_dDown_T: [d_ff, d_model]; H: [m,d_ff], c:[m,d_model]
    y = []
    if cons["H"].numel():
        H = cons["H"].to(device, compute_dtype)        # [m, d_ff]
        C = cons["c"].to(device, compute_dtype)        # [m, d_model]
        sc= cons["sc"].to(device, compute_dtype).squeeze(-1)
        M = H @ delta_dDown_T.to(device, compute_dtype) # [m, d_model]
        yf= sc * (M * C).sum(dim=1)
        y.append(yf)
    return torch.cat(y, dim=0) if y else torch.zeros(0, device=device, dtype=compute_dtype)

@torch.no_grad()
def AT_times_y_ffn_down(y, cons, d_ff, d_model, device="cpu", compute_dtype=torch.float32):
    dDown_T = torch.zeros((d_ff, d_model), device=device, dtype=compute_dtype)
    if cons["H"].numel():
        m = cons["H"].shape[0]
        w = (y[:m] * cons["sc"].squeeze(-1).to(device)).unsqueeze(1)
        H = cons["H"].to(device, compute_dtype)    # [m,d_ff]
        C = cons["c"].to(device, compute_dtype)    # [m,d_model]
        dDown_T += H.T @ (w * C)
    return dDown_T

# ===== FFN Dense/Cholesky efficient solvers (faster than CG when m is small) =====

@torch.no_grad()
def ffn_down_dense_project(cons, task_dDown_T, lam=1e-4, device="cpu", compute_dtype=torch.float32):
    """FFN Down: explicit Hadamard Gram + Cholesky solver (exact; faster for small m)"""
    # cons["H"]: [m, d_ff], cons["c"]: [m, d_model], cons["sc"]:[m,1]
    H = cons["H"].to(device, compute_dtype)               # [m, d_ff]
    C = cons["c"].to(device, compute_dtype)               # [m, d_model]
    s = cons["sc"].to(device, compute_dtype).squeeze(-1)  # [m]

    m = H.size(0)
    if m == 0:
        return task_dDown_T, {"residual_norm": 0.0, "solver": "dense_skip", "m": 0, "iterations": 0}

    # Gram: G = (s s^T) ⊙ (H H^T) ⊙ (C C^T) + λI
    HH = H @ H.T              # [m,m]
    CC = C @ C.T              # [m,m]
    G  = (HH * CC) * (s[:,None] * s[None,:])  # Hadamard
    G  = G + lam * torch.eye(m, device=device, dtype=compute_dtype)

    # rhs = s * diag( (H @ Δ) @ C^T )
    Δ = task_dDown_T.to(device, compute_dtype)            # [d_ff, d_model]
    M = (H @ Δ)                                           # [m, d_model]
    rhs = s * (M * C).sum(dim=1)                          # [m]

    # solve (G z = rhs)
    try:
        L = torch.linalg.cholesky(G)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # [m]
    except RuntimeError as e:
        # Fallback to LU if Cholesky fails
        z = torch.linalg.solve(G, rhs)

    w = z * s                                                   # [m]

    # Δ_proj = Δ - A^T z ; A^T z = H^T @ (w[:,None] * C)
    dT_w = H.T @ (w[:, None] * C)                # [d_ff, d_model]
    dT_proj = Δ - dT_w

    # Residual ||A Δ_proj||
    M2   = (H @ dT_proj)                                     # [m, d_model]
    resid= (s * (M2 * C).sum(dim=1)).norm().item()

    return dT_proj.to(task_dDown_T.dtype), {
        "residual_norm": resid, "solver": "dense_cholesky", "m": m, "iterations": 1
    }

@torch.no_grad()
def ffn_gate_dense_project(cons, task_dGate, lam=1e-4, device="cpu", compute_dtype=torch.float32):
    """FFN Gate: explicit Hadamard Gram + Cholesky solver"""
    X = cons["X_gate"].to(device, compute_dtype)               # [m, d_model]
    C = cons["c_gate"].to(device, compute_dtype)               # [m, d_ff]
    s = cons["sc_gate"].to(device, compute_dtype).squeeze(-1)  # [m]

    m = X.size(0)
    if m == 0:
        return task_dGate, {"residual_norm": 0.0, "solver": "dense_skip", "m": 0, "iterations": 0}

    # Gram: G = (s s^T) ⊙ (C C^T) ⊙ (X X^T) + λI
    XX = X @ X.T              # [m,m]
    CC = C @ C.T              # [m,m]
    G  = (CC * XX) * (s[:,None] * s[None,:])  # Hadamard
    G  = G + lam * torch.eye(m, device=device, dtype=compute_dtype)

    # rhs = s * ((X @ Δ^T) ⊙ C).sum(-1)
    Δ = task_dGate.to(device, compute_dtype)            # [d_ff, d_model]
    M = X @ Δ.T                                         # [m, d_ff]
    rhs = s * (M * C).sum(dim=1)                        # [m]

    # solve (G z = rhs)
    try:
        L = torch.linalg.cholesky(G)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # [m]
    except RuntimeError:
        z = torch.linalg.solve(G, rhs)

    w = z * s                                                   # [m]

    # Δ_proj = Δ - A^T z ; A^T z = C^T @ (w[:,None] * X)
    dG_w = C.T @ (w[:, None] * X)                # [d_ff, d_model]
    dG_proj = Δ - dG_w

    # Residual ||A Δ_proj||
    M2   = X @ dG_proj.T                                     # [m, d_ff]
    resid= (s * (M2 * C).sum(dim=1)).norm().item()

    return dG_proj.to(task_dGate.dtype), {
        "residual_norm": resid, "solver": "dense_cholesky", "m": m, "iterations": 1
    }


@torch.no_grad()
def ffn_up_dense_project(cons, task_dUp, lam=1e-4, device="cpu", compute_dtype=torch.float32):
    """FFN Up: explicit Hadamard Gram + Cholesky solver"""
    X = cons["X_up"].to(device, compute_dtype)               # [m, d_model]
    C = cons["c_up"].to(device, compute_dtype)               # [m, d_ff]
    s = cons["sc_up"].to(device, compute_dtype).squeeze(-1)  # [m]

    m = X.size(0)
    if m == 0:
        return task_dUp, {"residual_norm": 0.0, "solver": "dense_skip", "m": 0, "iterations": 0}

    # Gram: G = (s s^T) ⊙ (C C^T) ⊙ (X X^T) + λI
    XX = X @ X.T              # [m,m]
    CC = C @ C.T              # [m,m]
    G  = (CC * XX) * (s[:,None] * s[None,:])  # Hadamard
    G  = G + lam * torch.eye(m, device=device, dtype=compute_dtype)

    # rhs = s * ((X @ Δ^T) ⊙ C).sum(-1)
    Δ = task_dUp.to(device, compute_dtype)              # [d_ff, d_model]
    M = X @ Δ.T                                         # [m, d_ff]
    rhs = s * (M * C).sum(dim=1)                        # [m]

    # solve (G z = rhs)
    try:
        L = torch.linalg.cholesky(G)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # [m]
    except RuntimeError:
        z = torch.linalg.solve(G, rhs)

    w = z * s                                                   # [m]

    # Δ_proj = Δ - A^T z ; A^T z = C^T @ (w[:,None] * X)
    dU_w = C.T @ (w[:, None] * X)                # [d_ff, d_model]
    dU_proj = Δ - dU_w

    # Residual ||A Δ_proj||
    M2   = X @ dU_proj.T                                     # [m, d_ff]
    resid= (s * (M2 * C).sum(dim=1)).norm().item()

    return dU_proj.to(task_dUp.dtype), {
        "residual_norm": resid, "solver": "dense_cholesky", "m": m, "iterations": 1
    }


def cg_ffn_down(cons, task_dDown_T, lam=1e-4, maxit=100, tol=1e-5, device="cpu", compute_dtype=torch.float32):
    # Convert task_dDown_T to compute_dtype for CG
    task_dDown_T_compute = task_dDown_T.to(device, compute_dtype)
    rhs = A_times_delta_ffn_down(task_dDown_T_compute, cons, device, compute_dtype)
    if rhs.numel()==0:
        return task_dDown_T, {"rhs":rhs.cpu(),"z":torch.tensor([]),"residual_norm":0.0,"iterations":0}
    def Mv(z):
        dT = AT_times_y_ffn_down(z, cons, task_dDown_T_compute.size(0), task_dDown_T_compute.size(1), device, compute_dtype)
        Az = A_times_delta_ffn_down(dT, cons, device, compute_dtype)
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
    dT_w = AT_times_y_ffn_down(x, cons, task_dDown_T_compute.size(0), task_dDown_T_compute.size(1), device, compute_dtype)
    dT_proj = task_dDown_T_compute - dT_w
    res = A_times_delta_ffn_down(dT_proj, cons, device, compute_dtype)
    # Back to original dtype
    return dT_proj.to(task_dDown_T.dtype), {"rhs":rhs.cpu(),"z":x.cpu(),"residual_norm":res.norm().item(),"iterations":it+1}
