from __future__ import annotations

"""Facies-adaptive anisotropic reliability propagation (FARP).

Core idea
---------
We propagate a reliability field R(x) from sparse well seeds in an anisotropic
metric space. The metric is modulated by:
  1) channel-likeness p_ch(x)  (from facies label OR facies probability)
  2) waveform/attribute similarity (from seismic amplitude + gradient)
  3) local dominant orientation (structure tensor of p_ch)

This module supports BOTH:
  - synthetic / labeled case (Stanford VI-E): facies_3d given (hard labels)
  - iterative / pseudo-label case: provide p_channel_3d and conf_3d (or facies_prob_3d)

In iterative coupling, the recommended pattern is:
  p_ch := mix( prior_facies , predicted_facies , alpha ) with confidence gating,
  then R := EMA(R_prev, R_new).
"""

import torch
import torch.nn.functional as F


def _sobel_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sobel gradients for 2D tensors.

    x: [B,1,H,W]
    returns gx, gy: [B,1,H,W]
    """
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return gx, gy


def _gauss_blur(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Separable Gaussian blur with a small fixed kernel."""
    if sigma <= 0:
        return x
    radius = int(3 * sigma + 0.5)
    size = 2 * radius + 1
    coords = torch.arange(size, device=x.device, dtype=x.dtype) - radius
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / (g.sum() + 1e-12)
    g1 = g.view(1, 1, 1, size)
    g2 = g.view(1, 1, size, 1)
    x = F.conv2d(x, g1.expand(x.size(1), 1, 1, size), padding=(0, radius), groups=x.size(1))
    x = F.conv2d(x, g2.expand(x.size(1), 1, size, 1), padding=(radius, 0), groups=x.size(1))
    return x


def structure_tensor_orientation(p_channel: torch.Tensor, sigma: float = 1.2, eps: float = 1e-8) -> torch.Tensor:
    """Compute local dominant direction from structure tensor of p_channel.

    p_channel: [B,1,H,W] (float, 0..1)
    returns v: [B,2,H,W] unit vector (vx, vy)
    """
    gx, gy = _sobel_xy(p_channel)
    j11 = _gauss_blur(gx * gx, sigma)
    j22 = _gauss_blur(gy * gy, sigma)
    j12 = _gauss_blur(gx * gy, sigma)

    angle = 0.5 * torch.atan2(2 * j12, (j11 - j22 + eps))
    vx = torch.cos(angle)
    vy = torch.sin(angle)
    v = torch.cat([vx, vy], dim=1)
    v = v / (torch.sqrt((v * v).sum(dim=1, keepdim=True)) + eps)
    return v


def _neighbor_shift_8(x: torch.Tensor) -> list[torch.Tensor]:
    """8-neighborhood shifts for [B,C,H,W]."""
    E = F.pad(x[..., :, 1:], (0, 1, 0, 0))
    W = F.pad(x[..., :, :-1], (1, 0, 0, 0))
    S = F.pad(x[..., 1:, :], (0, 0, 0, 1))
    N = F.pad(x[..., :-1, :], (0, 0, 1, 0))
    SE = F.pad(x[..., 1:, 1:], (0, 1, 0, 1))
    SW = F.pad(x[..., 1:, :-1], (1, 0, 0, 1))
    NE = F.pad(x[..., :-1, 1:], (0, 1, 1, 0))
    NW = F.pad(x[..., :-1, :-1], (1, 0, 1, 0))
    return [E, W, S, N, SE, SW, NE, NW]


@torch.no_grad()
def anisotropic_reliability_2d(
    well_mask: torch.Tensor,  # [B,1,H,W]
    p_channel: torch.Tensor,  # [B,1,H,W]
    feat: torch.Tensor | None = None,  # [B,C,H,W]
    steps: int = 25,
    eta: float = 0.6,
    rho_aniso_map: torch.Tensor | None = None,  # [B,1,H,W], controls anisotropy only
    eta_update_map: torch.Tensor | None = None,  # [B,1,H,W], controls update amplitude only
    eta_map: torch.Tensor | None = None,  # [B,1,H,W] optional local step strength in [0,1]
    rho_skip: float = 0.10,
    gamma: float = 8.0,
    tau: float = 0.6,
    kappa: float = 4.0,
    sigma_st: float = 1.2,
    curr_epoch: int | None = None,
    max_epoch: int | None = None,
    gamma_warmup_ratio: float = 0.30,
    gamma_cap_ratio: float = 0.80,
    damp: torch.Tensor | None = None,  # [B,1,H,W] optional, 0..1, smaller -> harder to propagate
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate reliability on a 2D grid slice."""
    v = structure_tensor_orientation(p_channel, sigma=sigma_st, eps=eps)  # [B,2,H,W]
    R = well_mask.clone()

    # Decoupled maps:
    # - rho_aniso controls only anisotropic candidate interpolation
    # - alpha controls only update amplitude / skip
    if rho_aniso_map is not None:
        rho_aniso = rho_aniso_map.clamp(0.0, 1.0)
    elif eta_map is not None:
        rho_aniso = (eta_map / (float(eta) + eps)).clamp(0.0, 1.0)
    else:
        rho_aniso = torch.ones_like(well_mask)

    if eta_update_map is not None:
        alpha_map = (eta_update_map / (float(eta) + eps)).clamp(0.0, 1.0)
    elif eta_map is not None:
        alpha_map = (eta_map / (float(eta) + eps)).clamp(0.0, 1.0)
    else:
        alpha_map = torch.ones_like(well_mask)
    rho_aniso = rho_aniso.to(dtype=well_mask.dtype)
    alpha_map = alpha_map.to(dtype=well_mask.dtype)
    if (curr_epoch is None) or (max_epoch is None):
        gamma_t = float(gamma)
    else:
        p0 = float(gamma_warmup_ratio)
        denom = max(float(eps), (1.0 - p0))
        progress = float(curr_epoch) / float(max(1, int(max_epoch)))
        if progress < p0:
            gamma_t = 0.0
        else:
            gamma_t = float(gamma) * ((progress - p0) / denom)
    gamma_cap = float(gamma) * float(gamma_cap_ratio)
    gamma_t = min(float(gamma_t), float(gamma_cap))
    gamma_t = max(0.0, float(gamma_t))

    # 8 unit directions
    dirs = torch.tensor(
        [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]],
        device=well_mask.device,
        dtype=well_mask.dtype,
    )
    dirs = dirs / (torch.sqrt((dirs * dirs).sum(dim=1, keepdim=True)) + eps)

    def _propagate_once(R_in: torch.Tensor, gamma_field: torch.Tensor | float, use_aniso_dir: bool) -> torch.Tensor:
        Rn = _neighbor_shift_8(R_in)
        vx, vy = v[:, 0:1], v[:, 1:2]
        if isinstance(gamma_field, torch.Tensor):
            gk = torch.sigmoid(gamma_field * (p_channel - 0.5))
        else:
            gk = torch.sigmoid(float(gamma_field) * (p_channel - 0.5))
        if damp is not None:
            gk = gk * damp.clamp(0.0, 1.0)

        # orientation affinity (disabled for isotropic candidate)
        a_list = []
        for d in dirs:
            if use_aniso_dir:
                cos = d[0] * vx + d[1] * vy
                a_list.append(torch.exp(kappa * (cos * cos)))
            else:
                a_list.append(torch.ones_like(gk))

        # similarity affinity
        if feat is not None:
            Fn = _neighbor_shift_8(feat)
            s_list = []
            for fnb in Fn:
                dist = torch.sqrt(((feat - fnb) ** 2).sum(dim=1, keepdim=True) + eps)
                s_list.append(torch.exp(-dist / (tau + eps)))
        else:
            s_list = [None] * 8

        w_sum = torch.zeros_like(gk)
        w_list = []
        for i in range(8):
            s = s_list[i] if s_list[i] is not None else torch.ones_like(gk)
            w = gk * a_list[i] * s
            w_list.append(w)
            w_sum = w_sum + w
        w_sum = w_sum + eps

        R_prop_k = torch.zeros_like(R_in)
        for w, rnb in zip(w_list, Rn):
            R_prop_k = R_prop_k + (w / w_sum) * rnb
        return R_prop_k

    for _ in range(steps):
        R_in = R
        R_iso = _propagate_once(R_in, 0.0, use_aniso_dir=False)
        gamma_eff = float(gamma_t) * rho_aniso
        R_aniso = _propagate_once(R_in, gamma_eff, use_aniso_dir=True)
        R_prop = R_iso + rho_aniso * (R_aniso - R_iso)
        R_mix = (1.0 - alpha_map) * R_in + alpha_map * R_prop
        if float(rho_skip) > 0:
            mask = (alpha_map > float(rho_skip))
            R = torch.where(mask, R_mix, R_in)
        else:
            R = R_mix
        R = torch.maximum(R, well_mask)

    return R, v


def _ensure_probs(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """If x looks like logits, convert to probs; else assume already probs."""
    # Heuristic: if values outside [0,1] OR row-sum not ~1 => treat as logits.
    if x.numel() == 0:
        return x
    x_min = float(x.min())
    x_max = float(x.max())
    if (x_min < -1e-3) or (x_max > 1.0 + 1e-3):
        return torch.softmax(x, dim=dim)
    # already probs (best effort)
    return x


@torch.no_grad()
def build_R_and_prior_from_cube(
    seismic_3d: torch.Tensor,  # [H,IL,XL]
    ai_3d: torch.Tensor,       # [H,IL,XL] float
    well_trace_indices: torch.Tensor,  # [Nw] indices in flattened IL*XL order
    *,
    # --- channel-likeness sources (choose one) ---
    facies_3d: torch.Tensor | None = None,        # [H,IL,XL] int (hard labels)
    facies_prob_3d: torch.Tensor | None = None,   # [H,IL,XL,K] or [K,H,IL,XL] (prob OR logits)
    p_channel_3d: torch.Tensor | None = None,     # [H,IL,XL] float in [0,1]
    conf_3d: torch.Tensor | None = None,          # [H,IL,XL] float in [0,1]
    facies_prior_3d: torch.Tensor | None = None,  # [H,IL,XL] int (prior / anchor, e.g. from interpreter)
    # --- mixing / gating ---
    channel_id: int = 2,
    alpha_prior: float = 1.0,     # 1.0 => only prior/hard; 0.0 => only predicted
    conf_thresh: float = 0.75,    # below this, fall back to neutral/prior
    neutral_p: float = 0.5,       # fallback p_channel when no prior
    # --- anisotropic propagation ---
    steps_R: int = 25,
    eta: float = 0.6,
    rho_aniso_map_3d: torch.Tensor | None = None,   # [H,IL,XL], anisotropy-only control
    eta_update_map_3d: torch.Tensor | None = None,  # [H,IL,XL], update-only control
    eta_map_3d: torch.Tensor | None = None,        # [H,IL,XL], optional local adaptive eta(x)
    rho_skip: float = 0.10,
    gamma: float = 8.0,
    tau: float = 0.6,
    kappa: float = 4.0,
    sigma_st: float = 1.2,
    curr_epoch: int | None = None,
    max_epoch: int | None = None,
    gamma_warmup_ratio: float = 0.30,
    gamma_cap_ratio: float = 0.80,
    # --- physics damping (optional) ---
    phys_residual_3d: torch.Tensor | None = None,  # [H,IL,XL] >=0, larger => harder to propagate
    lambda_phys: float = 0.0,                      # 0 disables
    # --- optional soft impedance prior ---
    use_soft_prior: bool = False,
    steps_prior: int = 35,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build R(x) (and optional impedance prior) for the full cube.

    Returns:
      R_flat: [N, H] where N=IL*XL, values in [0,1]
      prior_flat (optional): [N,H]
    """
    device = seismic_3d.device
    H, IL, XL = seismic_3d.shape
    N = IL * XL

    # Backward-compatibility fallback for v2/legacy paths.
    if (rho_aniso_map_3d is None) and (eta_map_3d is not None):
        rho_aniso_map_3d = (eta_map_3d / (float(eta) + eps)).clamp(0.0, 1.0)
    if (eta_update_map_3d is None) and (eta_map_3d is not None):
        eta_update_map_3d = eta_map_3d

    # seeds: wells are entire traces (all depths) here
    well_mask_3d = torch.zeros((H, IL, XL), device=device, dtype=torch.float32)
    ii = (well_trace_indices // XL).long()
    jj = (well_trace_indices % XL).long()
    well_mask_3d[:, ii, jj] = 1.0

    # ---------- build p_channel and conf ----------
    # 1) prior channel probability (hard)
    if facies_prior_3d is not None:
        p_prior = (facies_prior_3d == channel_id).float()
    elif facies_3d is not None:
        p_prior = (facies_3d == channel_id).float()
    else:
        p_prior = None

    # 2) predicted / provided p_channel
    p_pred = None
    conf = None

    if p_channel_3d is not None:
        p_pred = p_channel_3d.float()
        conf = conf_3d.float() if conf_3d is not None else torch.ones_like(p_pred)
    elif facies_prob_3d is not None:
        fp = facies_prob_3d
        if fp.dim() == 4 and fp.shape[0] != H:
            # maybe [K,H,IL,XL] -> [H,IL,XL,K]
            fp = fp.permute(1, 2, 3, 0).contiguous()
        fp = _ensure_probs(fp.float(), dim=-1)
        p_pred = fp[..., channel_id]
        conf = fp.max(dim=-1).values
    elif facies_3d is not None:
        # fallback: hard labels as "pred"
        p_pred = (facies_3d == channel_id).float()
        conf = torch.ones_like(p_pred)

    if p_pred is None:
        raise ValueError("Need one of: p_channel_3d / facies_prob_3d / facies_3d.")

    # 3) mix + confidence gating
    if p_prior is not None:
        p_mix = alpha_prior * p_prior + (1.0 - alpha_prior) * p_pred
        p_fallback = alpha_prior * p_prior + (1.0 - alpha_prior) * float(neutral_p)
    else:
        p_mix = p_pred
        p_fallback = torch.full_like(p_pred, float(neutral_p))

    if conf is None:
        conf = torch.ones_like(p_pred)

    pch = torch.where(conf >= float(conf_thresh), p_mix, p_fallback).clamp(0.0, 1.0)

    # ---------- waveform/attribute embedding for similarity ----------
    amp = seismic_3d
    gx = F.pad(amp[:, :, 1:] - amp[:, :, :-1], (0, 1, 0, 0))
    gy = F.pad(amp[:, 1:, :] - amp[:, :-1, :], (0, 0, 0, 1))
    gmag = torch.sqrt(gx * gx + gy * gy + eps)
    feat = torch.stack([amp, gmag], dim=1)  # [H,2,IL,XL]

    # ---------- optional physics damping ----------
    if (phys_residual_3d is not None) and (lambda_phys > 0):
        res = phys_residual_3d.float().clamp(min=0.0)
        # robust normalization per depth slice (avoid being dominated by outliers)
        med = torch.median(res.reshape(H, -1), dim=1).values.view(H, 1, 1) + eps
        r = (res / med).clamp(0.0, 10.0)
        damp3d = torch.exp(-float(lambda_phys) * r).clamp(0.0, 1.0)  # 1 good, 0 bad
    else:
        damp3d = None

    # ---------- do 2D propagation per depth slice ----------
    R_out = torch.zeros((H, IL, XL), device=device, dtype=torch.float32)

    if use_soft_prior:
        prior = torch.zeros((H, IL, XL), device=device, dtype=torch.float32)
        prior[:, ii, jj] = ai_3d[:, ii, jj]
    else:
        prior = None

    for k in range(H):
        wm = well_mask_3d[k:k+1].unsqueeze(1)      # [1,1,IL,XL]
        pc = pch[k:k+1].unsqueeze(1)               # [1,1,IL,XL]
        fk = feat[k:k+1]                           # [1,2,IL,XL]
        dk = damp3d[k:k+1].unsqueeze(1) if damp3d is not None else None
        rak = rho_aniso_map_3d[k:k+1].unsqueeze(1) if rho_aniso_map_3d is not None else None
        euk = eta_update_map_3d[k:k+1].unsqueeze(1) if eta_update_map_3d is not None else None
        ek = eta_map_3d[k:k+1].unsqueeze(1) if eta_map_3d is not None else None

        Rk, _ = anisotropic_reliability_2d(
            wm, pc, fk,
            steps=steps_R,
            eta=eta,
            rho_aniso_map=rak,
            eta_update_map=euk,
            eta_map=ek,
            rho_skip=rho_skip,
            gamma=gamma,
            tau=tau,
            kappa=kappa,
            sigma_st=sigma_st,
            curr_epoch=curr_epoch,
            max_epoch=max_epoch,
            gamma_warmup_ratio=gamma_warmup_ratio,
            gamma_cap_ratio=gamma_cap_ratio,
            damp=dk,
        )
        R_out[k] = Rk[0, 0]

        if use_soft_prior and prior is not None:
            pk = prior[k:k+1].unsqueeze(1)  # [1,1,IL,XL]
            cond = Rk.clamp(0, 1)
            for _ in range(steps_prior):
                pn = _neighbor_shift_8(pk)
                condn = _neighbor_shift_8(cond)
                wsum = torch.zeros_like(pk)
                acc = torch.zeros_like(pk)
                for w, pnb in zip(condn, pn):
                    wsum = wsum + w
                    acc = acc + w * pnb
                pk = acc / (wsum + eps)
                pk = pk * (1 - wm) + prior[k:k+1].unsqueeze(1) * wm
            prior[k] = pk[0, 0]

    # flatten to [N,H]
    R_flat = R_out.reshape(H, N).transpose(0, 1).contiguous()
    if prior is not None:
        prior_flat = prior.reshape(H, N).transpose(0, 1).contiguous()
    else:
        prior_flat = None
    return R_flat, prior_flat
