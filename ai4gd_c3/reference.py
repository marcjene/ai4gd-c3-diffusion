"""Reference solutions of the C3-A8 exercises. Notation as in C3-A5 to C3-A7.

Schedule convention: beta, alpha, abar have length K + 1 and index 0 holds beta_0 = 0, alpha_0 = abar_0 = 1,
so that abar[k] is abar_k for k = 1..K.
"""
import torch
import torch.nn.functional as F


# Ex1
def normalise(pv_kw, nc_kw, capacity_kwp, max_export_kw):
    pv_pu = pv_kw / capacity_kwp
    nc_pu = nc_kw / max_export_kw
    return pv_pu, nc_pu


# Ex2
def linear_schedule(K, beta_1=1e-4, beta_K=0.2):
    beta = torch.cat([torch.zeros(1), torch.linspace(beta_1, beta_K, K)])
    alpha = 1 - beta
    abar = torch.cumprod(alpha, dim=0)
    return beta, alpha, abar


# Ex3
def q_sample(x0, k, eps, abar):
    a = abar[k].unsqueeze(-1)
    return a.sqrt() * x0 + (1 - a).sqrt() * eps


# Ex4a: output shape of each block of Fig. 3 in [1], batch B = 16, T = 48, 3 conditions, B_ch = 128
SHAPES = {"input": (16, 4, 48), "init_conv": (16, 128, 48), "encoder_1": (16, 256, 48),
          "encoder_2": (16, 512, 24), "bottleneck": (16, 512, 12), "decoder_2": (16, 256, 24),
          "decoder_1": (16, 128, 48), "output": (16, 48)}

# Ex4b: the complete class is ai4gd_c3.model.UNet1D (blanks: FiLM in ModulatedConv1D, skip concatenation)


# Ex4c
def diffusion_loss(model, x0, c, abar, K):
    k = torch.randint(1, K + 1, (x0.shape[0],), device=x0.device)
    eps = torch.randn_like(x0)
    x_k = q_sample(x0, k, eps, abar)
    return F.mse_loss(model(x_k, k, c), eps)


# Ex5
def reverse_step(x_k, eps_hat, k, beta, alpha, abar, z):
    mean = (x_k - beta[k] / (1 - abar[k]).sqrt() * eps_hat) / alpha[k].sqrt()
    return mean + beta[k].sqrt() * z if k > 1 else mean


# Ex5b
def predict_x0(x_k, eps_hat, k, abar):
    return (x_k - (1 - abar[k]).sqrt() * eps_hat) / abar[k].sqrt()


# Ex6
def postprocess(samples, ghi):
    return samples.clamp(min=0) * (ghi > 0)


# Ex7
def point_and_band(samples, level=80):
    q = torch.tensor([0.5, 0.5 - level / 200, 0.5 + level / 200], dtype=samples.dtype)
    median, lower, upper = torch.quantile(samples, q, dim=0)
    return median, lower, upper


def rmse(pv, pv_hat):
    return ((pv - pv_hat) ** 2).mean().sqrt()


# Ex8
def picp_pinaw(pv, lower, upper, ghi, capacity):
    sun = ghi > 0
    inside = (pv >= lower) & (pv <= upper)
    picp = 100 * inside[sun].float().mean()
    pinaw = ((upper - lower)[sun] / capacity).mean()
    return picp, pinaw


# Ex9
def pinball(pv, q_pred, q, ghi):
    err = pv - q_pred  # [Q, T]
    loss = torch.maximum(q[:, None] * err, (q[:, None] - 1) * err)
    return loss[:, ghi > 0].mean()


# Not an exercise: MASE as in C3-A7 (all 48 half-hours, naive = previous half-hour)
def mase(pv, pv_hat):
    n = len(pv)
    return (n - 1) / n * (pv - pv_hat).abs().sum() / (pv[1:] - pv[:-1]).abs().sum()
