"""Course model: the 1D U-Net of [1] (code_reproducibility/models/unet_v2.py at commit 1dd99af, same parameter
names, so the course checkpoints load), plus training (Algorithm 1) and sampling (Algorithm 2) loops.

Two interface changes from the paper code, none in the maths:
- forward(x_k, k, c) takes the step k = 1..K and passes k - 1 to the time embedding (the checkpoints were trained
  with 0-based steps), and c as one tensor [B, n_c, T] instead of a list.
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import DEVICE
from .data import fetch, train_set

K = 100
N_SAMPLES = 50 if DEVICE != "cpu" else 20  # samples per day; fewer without a GPU to keep the notebook usable
CONDITIONS = {"all": [0, 1, 2], "ghi": [0], "nc": [1, 2]}  # channels of c = (GHI_d, NC_d, NC_d-1)
HPARAMS = dict(base_channels=128, time_embed_dim=32, diffusion_steps=K, depth=2)


def get_sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]  # [B, half_dim]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, dim]


class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(nn.Linear(embed_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, t):
        return self.mlp(get_sinusoidal_embedding(t, dim=self.embed_dim))  # [B, out_dim]


class ModulatedConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.film = nn.Linear(time_emb_dim, out_ch * 2)

    def forward(self, x, t_emb):
        h = self.conv(x)
        scale, shift = self.film(t_emb).chunk(2, dim=-1)  # FiLM: one (scale, shift) per channel
        return F.relu(h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1))


class UNet1D(nn.Module):
    def __init__(self, in_ch: int, base_channels: int, time_embed_dim: int, diffusion_steps: int, depth: int = 2):
        super().__init__()
        self.time_embedding = TimestepEmbedding(time_embed_dim, base_channels)
        self.init_conv = nn.Conv1d(in_ch, base_channels, kernel_size=3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2)
        ch, skip_ch = base_channels, []
        for _ in range(depth):
            self.enc_blocks.append(ModulatedConv1D(ch, ch * 2, base_channels))
            skip_ch.append(ch)
            ch *= 2
        self.bot1 = ModulatedConv1D(ch, ch * 2, base_channels)
        self.bot2 = ModulatedConv1D(ch * 2, ch, base_channels)
        self.up_blocks, self.dec_blocks = nn.ModuleList(), nn.ModuleList()
        for s in reversed(skip_ch):
            self.up_blocks.append(nn.ConvTranspose1d(ch, s, kernel_size=2, stride=2))
            self.dec_blocks.append(ModulatedConv1D(ch + s, s, base_channels))
            ch = s
        self.final_conv = nn.Conv1d(ch, 1, kernel_size=3, padding=1)

    def forward(self, x, k, c):
        """x: x_k [B, T]; k: step 1..K [B]; c: conditions [B, n_c, T]. Returns eps_theta [B, T]."""
        t_emb = self.time_embedding(k - 1)  # the checkpoints use 0-based steps
        x = self.init_conv(torch.cat([x.unsqueeze(1), c], dim=1))  # input -> initial convolution
        skips = []
        for enc in self.enc_blocks:  # encoder levels 1, 2
            x = F.relu(enc(x, t_emb))
            skips.append(x)
            x = self.pool(x)
        x = F.relu(self.bot1(x, t_emb))  # bottleneck
        x = F.relu(self.bot2(x, t_emb))
        for up, dec in zip(self.up_blocks, self.dec_blocks):  # decoder levels 2, 1
            x = up(x)
            x = torch.cat([x, skips.pop()], dim=1)  # skip connection
            x = F.relu(dec(x, t_emb))
        return self.final_conv(x).squeeze(1)  # output


def new_model(conditions="all", cls=UNet1D):
    m = cls(len(CONDITIONS[conditions]) + 1, **HPARAMS).to(DEVICE)
    m.cond_idx = CONDITIONS[conditions]
    return m


def load_pretrained(conditions="all", cls=UNet1D):
    """Course checkpoint pv_<conditions>.pt (course schedule, seed 0, commit 1dd99af)."""
    m = new_model(conditions, cls)
    m.load_state_dict(torch.load(fetch(f"pv_{conditions}.pt"), map_location=DEVICE))
    return m.eval()


def train(model, loss_fn, abar, steps=2000, batch_size=256, lr=1e-3, seed=0):
    """Algorithm 1: `steps` gradient steps on random mini-batches of training days. Returns the losses."""
    torch.manual_seed(seed)
    x0_all, c_all = (t.to(DEVICE) for t in train_set())
    c_all = c_all[:, getattr(model, "cond_idx", CONDITIONS["all"])]
    abar = abar.to(DEVICE)
    K = len(abar) - 1  # index 0 holds abar_0 = 1
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses, t0 = [], time.time()
    for step in range(steps):
        i = torch.randint(0, len(x0_all), (batch_size,), device=DEVICE)
        loss = loss_fn(model, x0_all[i], c_all[i], abar, K)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 250 == 0 or step == steps - 1:
            print(f"step {step:4d}  loss {loss.item():.4f}  ({time.time() - t0:.0f} s)")
    model.eval()
    return losses


@torch.no_grad()
def sample(model, c, schedule, step_fn, n=None, seed=0, record=()):
    """Algorithm 2: n samples per day (default N_SAMPLES), from x_K ~ N(0, I) down to x_0. Raw output.

    c: [3, T] (one day) or [D, 3, T]; returns [n, T] or [D, n, T].
    record: steps k whose (x_k, eps_hat) are kept, returned as a dict {k: (x_k, eps_hat)} next to x_0.
    """
    beta, alpha, abar = (s.to(DEVICE) for s in schedule)
    K = len(abar) - 1
    n = n or N_SAMPLES
    one_day = c.dim() == 2
    c = (c[None] if one_day else c)[:, getattr(model, "cond_idx", CONDITIONS["all"])].to(DEVICE)
    D = c.shape[0]
    c = c.repeat_interleave(n, 0)
    torch.manual_seed(seed)
    x = torch.randn(D * n, c.shape[-1], device=DEVICE)
    kept = {}
    for k in range(K, 0, -1):
        eps_hat = model(x, torch.full((D * n,), k, device=DEVICE), c)
        if k in record:
            kept[k] = (x.view(D, n, -1).cpu(), eps_hat.view(D, n, -1).cpu())
        z = torch.randn_like(x) if k > 1 else torch.zeros_like(x)
        x = step_fn(x, eps_hat, k, beta, alpha, abar, z)
    x = x.view(D, n, -1).cpu()
    if one_day:
        x, kept = x[0], {k: (a[0], b[0]) for k, (a, b) in kept.items()}
    return (x, kept) if record else x
