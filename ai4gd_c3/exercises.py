"""Exercise checks, hints, solutions, questions and progress for C3-A7 (Kaggle-Learn style).

check() never raises: it prints "✅ Correct, you can continue." or one diagnosis. Diagnoses come from comparing the
learner's output with known wrong variants of the reference on fixed random inputs.
"""
import inspect
import json
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import DEVICE, data, model, reference as ref

_STATE = Path.home() / ".ai4gd_c3_progress.json"
__all__ = ["____", "ex1", "ex2", "ex3", "ex4a", "ex4b", "ex4c", "ex5", "ex5b", "ex6", "ex7", "ex8", "ex9",
           "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "progress", "catch_up"]
OK = "✅ Correct, you can continue."


class BlankError(Exception):
    pass


class Blank:
    """The ____ placeholder: any use of it raises BlankError."""
    def _fail(self, *a, **k):
        raise BlankError("____")
    __getattr__ = __call__ = __getitem__ = __iter__ = __bool__ = __float__ = __index__ = _fail
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _fail
    __pow__ = __rpow__ = __neg__ = __lt__ = __gt__ = __le__ = __ge__ = __and__ = __rand__ = _fail

    def __repr__(self):
        return "____"


____ = Blank()


def _load_state():
    try:
        return json.loads(_STATE.read_text())
    except Exception:  # missing or unreadable file: start empty
        return {}


def _save(name, ok):
    s = _load_state()
    s[name] = bool(ok) or s.get(name, False)  # once solved, stays solved
    try:
        _STATE.write_text(json.dumps(s))
    except OSError:
        pass


# ---------------------------------------------------------------- comparison helpers
def _t(x):
    return torch.as_tensor(x, dtype=torch.float32).detach().cpu() if not isinstance(x, Blank) else x


def _close(a, b, atol=1e-5):
    a, b = _t(a), _t(b)
    return a.shape == b.shape and torch.allclose(a, b, atol=atol, rtol=1e-4)


def _as_tuple(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def _compare(fn, ref_fn, cases, variants=(), names=None, atol=1e-5):
    """None if fn matches ref_fn on every case, else a diagnosis string.

    variants: (message, wrong_fn, index): wrong_fn returns a full (wrong) output; index restricts the variant to
    the output element it explains (None: any element).
    """
    for args in cases:
        exp = _as_tuple(ref_fn(*args))
        got = fn(*[a.clone() if torch.is_tensor(a) else a for a in args])
        if got is None:
            return "your function returns None: is the `return` line filled in?"
        got = _as_tuple(got)
        if any(isinstance(g, Blank) for g in got):
            raise BlankError
        if len(got) != len(exp):
            return f"expected {len(exp)} outputs, got {len(got)}."
        for i, (g, e) in enumerate(zip(got, exp)):
            if _close(g, e, atol):
                continue
            for msg, wrong, idx in variants:
                if idx not in (None, i):
                    continue
                try:
                    w = _as_tuple(wrong(*args))
                    if _close(g, w[i], atol):
                        return msg
                except Exception:  # a variant may not apply to this case
                    pass
            name = names[i] if names else "the output"
            g, e = _t(g), _t(e)
            if g.shape != e.shape:
                return f"{name} has shape {tuple(g.shape)}, expected {tuple(e.shape)}."
            j = (g - e).abs().flatten().argmax().item()
            return f"{name} is wrong: at flat index {j} you return {g.flatten()[j]:.5g}, expected {e.flatten()[j]:.5g}."
    return None


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------- exercise and question objects
class Exercise:
    def __init__(self, name, title, check, refs, hint, errors=(), optional=False, show=None):
        self.name, self.title, self._check, self._refs, self._show = name, title, check, refs, show or refs
        self._hint, self._errors, self.optional = hint, errors, optional

    def _run(self, *args):
        try:
            return self._check(*args)
        except BlankError:
            return "⬜ not started yet: replace every ____ with your code."
        except Exception as e:  # never raise: map the error to a diagnosis
            if "____" in str(e):
                return "⬜ not started yet: replace every ____ with your code."
            for text, msg in self._errors:
                if text in str(e):
                    return msg
            return f"your code raised {type(e).__name__}: {str(e).splitlines()[0][:200]}"

    def check(self, *args):
        msg = self._run(*args)
        _save(self.name, msg is None)
        print(OK if msg is None else msg if msg.startswith("⬜") else f"❌ {msg}  (stuck? {self.name}.hint())")

    def use(self, *args):
        """Your function(s) if they pass the check, otherwise the reference (with a one-line notice)."""
        if self._run(*args) is None:
            return args[0] if len(args) == 1 else args
        print(f"↪ {self.name}: using the reference solution, so that the notebook can continue.")
        r = [getattr(ref, n) if isinstance(n, str) else n for n in self._refs]
        return r[0] if len(r) == 1 else tuple(r)

    def hint(self):
        print(f"💡 {self._hint}")

    def solution(self):
        for r in self._show:
            print(inspect.getsource(r) if not isinstance(r, str) else f"{r} = {getattr(ref, r)!r}")


class Question:
    def __init__(self, name, answer, feedback, optional=False):
        self.name, self.answer, self.feedback, self.optional = name, answer, feedback, optional
        self.title = name

    def check(self, choice):
        choice = str(choice).strip().upper()
        if choice not in self.feedback:
            print(f"Answer with one of {', '.join(self.feedback)}.")
            return
        ok = choice == self.answer
        _save(self.name, ok)
        print(("✅ " if ok else "❌ Not quite. ") + self.feedback[choice])


# ---------------------------------------------------------------- Ex1 normalise
def _check1(fn):
    g = _gen()
    cases = []
    for _ in range(5):
        cap, mexp = 0.5 + 4 * torch.rand(1, generator=g).item(), 0.3 + 3 * torch.rand(1, generator=g).item()
        pv = cap * torch.rand(48, generator=g)
        nc = torch.randn(48, generator=g) * mexp
        cases.append((pv, nc, cap, mexp))
    mm = lambda x: (x - x.min()) / (x.max() - x.min())  # noqa: E731
    return _compare(fn, ref.normalise, cases, names=["pv_pu", "nc_pu"], variants=[
        ("NC is divided by the household's maximum export, not by the PV capacity.",
         lambda p, n, c, m: (p / c, n / c), 1),
        ("you swapped the two divisors: PV ÷ capacity, NC ÷ maximum export.", lambda p, n, c, m: (p / m, n / c), None),
        ("min–max scaling moves exports above zero; divide by a physical value instead (C3-A5).",
         lambda p, n, c, m: (mm(p), mm(n)), None),
        ("min–max scaling moves exports above zero; divide NC by the maximum export (C3-A5).",
         lambda p, n, c, m: (p / c, mm(n)), 1),
        ("multiplying goes from pu back to kW; here we go from kW to pu.", lambda p, n, c, m: (p * c, n * m), None),
        ("multiplying goes from pu back to kW; here we go from kW to pu.", lambda p, n, c, m: (p * c, n / m), 0),
        ("multiplying goes from pu back to kW; here we go from kW to pu.", lambda p, n, c, m: (p / c, n * m), 1),
    ])


# ---------------------------------------------------------------- Ex2 schedule
def _check2(fn):
    def unpadded(K, b1=1e-4, bK=0.2):
        b = torch.linspace(b1, bK, K)
        return b, 1 - b, torch.cumprod(1 - b, 0)
    cases = [(10,), (100,), (1000,), (100, 1e-4, 0.05)]
    msg = _compare(fn, ref.linear_schedule, cases, names=["beta", "alpha", "abar"], variants=[
        ("off by one: index 0 must hold beta_0 = 0 and ᾱ_0 = 1, so that abar[k] is ᾱ_k (length K + 1).",
         unpadded, None),
        ("that is the paper's schedule (β_K = 0.02); the course uses β_K = 0.2.",
         lambda K, b1=1e-4, bK=0.2: ref.linear_schedule(K, b1, 0.02), None),
        ("use the arguments beta_1 and beta_K instead of fixed numbers.",
         lambda K, b1=1e-4, bK=0.2: ref.linear_schedule(K), None),
        ("ᾱ_k is the cumulative product of α_1 … α_k (torch.cumprod), not α_k itself.",
         lambda K, *a: (lambda b, al, ab: (b, al, al))(*ref.linear_schedule(K, *a)), 2),
        ("ᾱ_k is a cumulative product, not a cumulative sum.",
         lambda K, *a: (lambda b, al, ab: (b, al, torch.cumsum(al, 0)))(*ref.linear_schedule(K, *a)), 2),
        ("you took the product of β instead of α = 1 − β.",
         lambda K, *a: (lambda b, al, ab: (b, al, torch.cumprod(b, 0)))(*ref.linear_schedule(K, *a)), 2),
    ])
    return msg


# ---------------------------------------------------------------- Ex3 closed-form x_k
def _cases3():
    g = _gen(3)
    abar = ref.linear_schedule(100)[2]
    out = []
    for _ in range(4):
        k = torch.randint(1, 101, (7,), generator=g)
        k[0], k[1] = 1, 100
        out.append((torch.rand(7, 48, generator=g), k, torch.randn(7, 48, generator=g), abar))
    return out


def _check3(fn):
    A = lambda abar, k: abar[k].unsqueeze(-1)  # noqa: E731
    return _compare(fn, ref.q_sample, _cases3(), names=["x_k"], variants=[
        ("√ missing: the shares are √ᾱ_k and √(1 − ᾱ_k).",
         lambda x, k, e, ab: A(ab, k) * x + (1 - A(ab, k)) * e, None),
        ("off by one in k: with our index-0 convention, ᾱ_k is abar[k].",
         lambda x, k, e, ab: ref.q_sample(x, k - 1, e, ab), None),
        ("the signal share √ᾱ_k multiplies x_0 and the noise share √(1 − ᾱ_k) multiplies ε.",
         lambda x, k, e, ab: (1 - A(ab, k)).sqrt() * x + A(ab, k).sqrt() * e, None),
        ("√(1 − ᾱ_k), not 1 − √ᾱ_k.", lambda x, k, e, ab: A(ab, k).sqrt() * x + (1 - A(ab, k).sqrt()) * e, None),
    ])


_ERR_BROADCAST = ("must match the size of tensor",
                  "abar[k] has shape [B]: add a time axis with .unsqueeze(-1) (or [:, None]) so it broadcasts over T.")


# ---------------------------------------------------------------- Ex4a shapes
_SHAPE_HINT = {
    "input": "x_k is 1 channel and the 3 conditions are 3 more: 4 channels of 48.",
    "init_conv": "the initial convolution maps the input to B_ch = 128 channels; the length is unchanged.",
    "encoder_1": "each encoder level doubles the channels; level 1 still works at full length 48.",
    "encoder_2": "max pooling halves the length before encoder level 2, which doubles the channels again.",
    "bottleneck": "pooled once more to 12 values; the two convolutions go 512 → 1024 → 512.",
    "decoder_2": "upsampled to 24, joined with skip 2, then the modulated convolution reduces the channels to 256.",
    "decoder_1": "back to 48 values, joined with skip 1, then reduced to 128 channels.",
    "output": "the last convolution gives 1 channel, which is squeezed away: (B, T).",
}


def _check4a(shapes):
    if not isinstance(shapes, dict):
        return "give a dict {block name: shape}."
    missing, extra = set(ref.SHAPES) - set(shapes), set(shapes) - set(ref.SHAPES)
    if missing or extra:
        return f"keys missing: {sorted(missing)}; unexpected keys: {sorted(extra)}."
    for key, exp in ref.SHAPES.items():
        got = shapes[key]
        if isinstance(got, Blank) or (isinstance(got, (tuple, list)) and any(isinstance(v, Blank) for v in got)):
            raise BlankError
        got = tuple(int(v) for v in got)
        if got == exp:
            continue
        if got == exp[1:]:
            return f"{key}: include the batch dimension B = 16 first."
        if key == "output" and got == (16, 1, 48):
            return "output: the final .squeeze(1) removes the channel axis."
        return f"{key}: expected {exp}, you gave {got}. {_SHAPE_HINT[key]}"
    return None


# ---------------------------------------------------------------- Ex4b complete the U-Net
def _variant_model(kind):
    m = model.load_pretrained("all")

    def film_off(self, x, t_emb):
        return F.relu(self.conv(x))

    def scale_only(self, x, t_emb):
        s, b = self.film(t_emb).chunk(2, dim=-1)
        return F.relu(self.conv(x) * s.unsqueeze(-1) + b.unsqueeze(-1))
    for mod in m.modules():
        if isinstance(mod, model.ModulatedConv1D):
            mod.forward = types.MethodType({"film_off": film_off, "scale_only": scale_only}[kind], mod)
    return m


def _check4b(cls):
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        return "pass the class itself: ex4b.check(UNet1D)."
    try:
        mine = cls(4, **model.HPARAMS).to(DEVICE)
    except TypeError as e:
        return f"keep the constructor of the original code, UNet1D(in_ch, base_channels, time_embed_dim, diffusion_steps, depth): {e}"
    state = torch.load(data.fetch("pv_all.pt"), map_location=DEVICE)
    try:
        mine.load_state_dict(state)
    except RuntimeError as e:
        text = str(e)
        if "size mismatch" in text:
            line = next(ln for ln in text.splitlines() if "size mismatch" in ln)
            return f"the course weights do not fit a layer: {line.strip()} Check the channel numbers."
        return ("you renamed, added or removed a layer, so the course weights do not load. Keep the names of the "
                f"original code. Details: {text.splitlines()[1].strip()[:200]}")
    mine.eval()
    g = _gen(4)
    x, c = torch.randn(4, 48, generator=g).to(DEVICE), torch.rand(4, 3, 48, generator=g).to(DEVICE)
    k = torch.tensor([1, 10, 50, 100], device=DEVICE)
    with torch.no_grad():
        out = mine(x, k, c).cpu()
        if _close(out, model.load_pretrained("all")(x, k, c).cpu(), 1e-4):
            return None
        if _close(out, _variant_model("film_off")(x, k, c).cpu(), 1e-4):
            return "FiLM missing: the step k never reaches the block. Compute scale and shift from self.film(t_emb)."
        if _close(out, _variant_model("scale_only")(x, k, c).cpu(), 1e-4):
            return "use (1 + scale), so that a zero FiLM output leaves h unchanged (that is how the weights were trained)."
    return ("the output differs from the course model. Check the concatenation order ([x, skip], as in the original "
            "code), and that scale and shift get a time axis with .unsqueeze(-1).")


# ---------------------------------------------------------------- Ex4c loss of one training iteration
class _Spy(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.g = _gen(44)

    def forward(self, x_k, k, c):
        y = torch.randn(x_k.shape, generator=self.g).to(x_k.device)
        self.calls.append((x_k.detach(), torch.as_tensor(k).detach(), c, y))
        return y


def _check4c(fn):
    K, B = 100, 64
    abar = ref.linear_schedule(K)[2].to(DEVICE)
    g = _gen(5)
    x0, c = torch.rand(B, 48, generator=g).to(DEVICE), torch.rand(B, 3, 48, generator=g).to(DEVICE)
    spy, ks = _Spy(), []
    for i in range(300):
        loss = fn(spy, x0, c, abar, K)
        if isinstance(loss, Blank):
            raise BlankError
        if loss is None:
            return "your function returns None: is the `return` line filled in?"
        if len(spy.calls) != i + 1:
            return "call the model exactly once per iteration: model(x_k, k, c)."
        x_k, k, c_seen, y = spy.calls[-1]
        if k.dim() == 0 or k.numel() == 1 or k.shape[0] != B:
            return "draw one step k per profile: a tensor of shape [B]."
        if c_seen is not c and not torch.equal(c_seen, c):
            return "pass the conditions c unchanged: only x_0 is noised, the conditions stay clean."
        if (k < 1).any():
            return "k runs from 1 to K: you drew k = 0. randint's upper bound is exclusive, so torch.randint(1, K + 1, ...)."
        if (k > K).any():
            return "k runs from 1 to K: you drew k > K."
        ks.append(k.cpu())
        if i < 3:
            a = abar[k.to(DEVICE)].unsqueeze(-1)
            eps = (x_k - a.sqrt() * x0) / (1 - a).sqrt()
            loss = torch.as_tensor(loss)
            if loss.numel() != 1:
                return "return one number: the mean over the batch and the half-hours (F.mse_loss)."
            if abs(eps.mean().item()) > 0.15 or not 0.85 < eps.std().item() < 1.15:
                if 0.2 < eps.std().item() < 0.4:
                    return "ε must be Gaussian noise: torch.randn_like, not torch.rand_like (uniform)."
                if torch.allclose(x_k, x0):
                    return "x_k must be the noised profile from q_sample(x0, k, eps, abar), not x_0."
                return "x_k does not match √ᾱ_k·x_0 + √(1 − ᾱ_k)·ε with Gaussian ε (check q_sample and abar[k])."
            if torch.allclose(loss.cpu(), F.mse_loss(y, x0).cpu(), atol=1e-5):
                return "the network predicts the noise ε, not x_0: compare its output with eps."
            if not torch.allclose(loss.cpu(), F.mse_loss(y, eps).cpu(), atol=1e-4):
                return "the loss must be the mean of (ε − ε_θ(x_k, k, c))², e.g. F.mse_loss(model(x_k, k, c), eps)."
    if (torch.cat(ks) == K).sum() == 0:
        return "k never reached K: randint's upper bound is exclusive, so torch.randint(1, K + 1, ...)."
    return None


# ---------------------------------------------------------------- Ex5 one reverse step, Ex5b x0_hat
def _cases5():
    g = _gen(6)
    sched = ref.linear_schedule(100)
    return [(torch.randn(5, 48, generator=g), torch.randn(5, 48, generator=g), k, *sched,
             torch.randn(5, 48, generator=g)) for k in (2, 50, 100, 1)]


def _check5(fn):
    def v(m):  # variant of the mean, same noise rule as the reference
        return lambda x, e, k, b, a, ab, z: m(x, e, k, b, a, ab) + (b[k].sqrt() * z if k > 1 else 0)
    return _compare(fn, ref.reverse_step, _cases5(), names=["x_{k-1}"], variants=[
        ("no fresh noise at the last step (k = 1): return the mean only.",
         lambda x, e, k, b, a, ab, z: (x - b[k] / (1 - ab[k]).sqrt() * e) / a[k].sqrt() + b[k].sqrt() * z, None),
        ("√ missing: the noise is σ_k·z with σ_k = √β_k.",
         lambda x, e, k, b, a, ab, z: (x - b[k] / (1 - ab[k]).sqrt() * e) / a[k].sqrt() + (b[k] * z if k > 1 else 0),
         None),
        ("off by one in k: with our index-0 convention use beta[k], alpha[k], abar[k].",
         lambda x, e, k, b, a, ab, z: ref.reverse_step(x, e, k - 1, b, a, ab, z) if k > 2 else None, None),
        ("sign: subtract the predicted noise.", v(lambda x, e, k, b, a, ab: (x + b[k] / (1 - ab[k]).sqrt() * e) / a[k].sqrt()), None),
        ("√α_k in the denominator (√ missing).", v(lambda x, e, k, b, a, ab: (x - b[k] / (1 - ab[k]).sqrt() * e) / a[k]), None),
        ("√(1 − ᾱ_k) under β_k (√ missing).", v(lambda x, e, k, b, a, ab: (x - b[k] / (1 - ab[k]) * e) / a[k].sqrt()), None),
    ])


def _check5b(fn):
    g = _gen(7)
    abar = ref.linear_schedule(100)[2]
    cases = [(torch.randn(5, 48, generator=g), torch.randn(5, 48, generator=g), k, abar) for k in (1, 30, 80, 100)]
    return _compare(fn, ref.predict_x0, cases, names=["x0_hat"], variants=[
        ("√ missing: x̂_0 = (x_k − √(1 − ᾱ_k)·ε̂) / √ᾱ_k.", lambda x, e, k, ab: (x - (1 - ab[k]) * e) / ab[k], None),
        ("√ missing on ᾱ_k in the denominator.", lambda x, e, k, ab: (x - (1 - ab[k]).sqrt() * e) / ab[k], None),
        ("off by one in k: use abar[k].", lambda x, e, k, ab: ref.predict_x0(x, e, k - 1, ab), None),
        ("sign: subtract the noise share.", lambda x, e, k, ab: (x + (1 - ab[k]).sqrt() * e) / ab[k].sqrt(), None),
    ])


# ---------------------------------------------------------------- Ex6 post-processing
def _check6(fn):
    g = _gen(8)
    ghi = torch.zeros(48)
    ghi[12:38] = torch.linspace(0, 3.14, 26).sin() * 0.9 + 0.001
    cases = [(torch.randn(6, 48, generator=g) * 0.4 + 0.2, ghi) for _ in range(3)]
    for s, gh in cases:
        before = s.clone()
        fn(s, gh)
        if not torch.equal(before, s):
            return "don't modify `samples` in place (clamp_, *=, s[...] = ...): return a new tensor."
    return _compare(fn, ref.postprocess, cases, names=["samples"], variants=[
        ("set PV to 0 where GHI = 0 (night): multiply by the mask (ghi > 0).", lambda s, gh: s.clamp(min=0), None),
        ("clamp the samples at 0: PV cannot be negative.", lambda s, gh: s * (gh > 0), None),
        ("mask only where GHI is exactly 0, not below a threshold.", lambda s, gh: s.clamp(min=0) * (gh > 0.05), None),
    ])


# ---------------------------------------------------------------- Ex7 median, PI band, RMSE
def _check7(fn, rmse_fn=None):
    g = _gen(9)
    cases = [(torch.rand(50, 48, generator=g) ** 2, lvl) for lvl in (80, 50, 90)]
    lo_hi = lambda s, l: (torch.quantile(s, 1 - l / 100, 0), torch.quantile(s, l / 100, 0))  # noqa: E731
    msg = _compare(fn, ref.point_and_band, cases, names=["median", "lower", "upper"], variants=[
        ("quantiles over the samples (dim=0), so that you get one value per half-hour.",
         lambda s, l: ref.point_and_band(s.T, l), None),
        ("the point estimate here is the median (quantile 0.5), not the mean.",
         lambda s, l: (s.mean(0), *ref.point_and_band(s, l)[1:]), 0),
        ("the 80 % PI runs from the 0.1 to the 0.9 quantile, not from 0.2 to 0.8: use 0.5 ∓ level/200.",
         lambda s, l: (ref.point_and_band(s, l)[0], *lo_hi(s, l)), None),
        ("lower and upper are swapped.", lambda s, l: (lambda m, lo, up: (m, up, lo))(*ref.point_and_band(s, l)), None),
    ])
    if msg or rmse_fn is None:
        return msg or "⬜ point_and_band is correct; now check both: ex7.check(point_and_band, rmse)."
    rc = [(torch.rand(48, generator=g), torch.rand(48, generator=g)) for _ in range(4)]
    return _compare(rmse_fn, ref.rmse, rc, names=["RMSE"], variants=[
        ("that is the MSE: √ missing.", lambda a, b: ((a - b) ** 2).mean(), None),
        ("that is the mean absolute error: square the errors, average, then take √.", lambda a, b: (a - b).abs().mean(), None),
        ("average over the half-hours (mean), not the sum.", lambda a, b: ((a - b) ** 2).sum().sqrt(), None),
    ])


_ERR_QUANTILE = ("q values must be in the range [0, 1]", "quantile levels are fractions: 0.1, not 10.")


# ---------------------------------------------------------------- Ex8 PICP and PINAW
def _cases8():
    g = _gen(10)
    ghi = torch.zeros(48)
    ghi[12:38] = torch.linspace(0.01, 1, 26)
    out = []
    for cap in (1.48, 3.2, 0.9):
        pv = cap * torch.rand(48, generator=g) * (ghi > 0)
        lower = (pv - 0.3 * cap * torch.rand(48, generator=g) + 0.1 * cap * torch.randn(48, generator=g)).clamp(min=0) * (ghi > 0)
        upper = lower + 0.4 * cap * torch.rand(48, generator=g) * (ghi > 0)
        out.append((pv, lower, upper, ghi, cap))
    return out


def _check8(fn):
    def unmasked(pv, lo, up, gh, cap):
        return 100 * ((pv >= lo) & (pv <= up)).float().mean(), ((up - lo) / cap).mean()
    return _compare(fn, ref.picp_pinaw, _cases8(), names=["PICP", "PINAW"], variants=[
        ("count only daylight half-hours (GHI > 0): at night every interval 'covers' the true zero (C3-A6).",
         unmasked, 0),
        ("average the width over daylight half-hours only (GHI > 0), as for PICP.", unmasked, 1),
        ("PICP is a percentage: multiply the share by 100.",
         lambda *a: (lambda p, w: (p / 100, w))(*ref.picp_pinaw(*a)), 0),
        ("PINAW is normalised by the installed capacity C (kW/kWp): divide the width by capacity.",
         lambda pv, lo, up, gh, cap: ref.picp_pinaw(pv, lo, up, gh, 1.0), 1),
        ("sign: the width is upper − lower.", lambda *a: (lambda p, w: (p, -w))(*ref.picp_pinaw(*a)), 1),
    ])


# ---------------------------------------------------------------- Ex9 pinball loss
def _check9(fn):
    g = _gen(11)
    q = torch.arange(1, 10) / 10
    ghi = torch.zeros(48)
    ghi[12:38] = 0.5
    cases = []
    for _ in range(3):
        pv = torch.rand(48, generator=g) * (ghi > 0)
        qp = torch.sort(torch.rand(9, 48, generator=g), 0).values * (ghi > 0)
        cases.append((pv, qp, q, ghi))
    return _compare(fn, ref.pinball, cases, names=["pinball loss"], variants=[
        ("q weights the case 'truth above the quantile' and 1 − q the case below: you swapped them.",
         lambda pv, qp, q, gh: ref.pinball(pv, qp, 1 - q, gh), None),
        ("average only over daylight half-hours (GHI > 0).", lambda pv, qp, q, gh: ref.pinball(pv, qp, q, torch.ones(48)), None),
        ("that is the mean absolute error; the pinball loss weights errors by q and 1 − q.",
         lambda pv, qp, q, gh: (pv - qp)[:, gh > 0].abs().mean(), None),
        ("take the mean over quantiles and half-hours, not the sum.",
         lambda pv, qp, q, gh: ref.pinball(pv, qp, q, gh) * 9 * 26, None),
    ])


# ---------------------------------------------------------------- the exercises
ex1 = Exercise("ex1", "Ex1 normalise PV and NC", _check1, [ref.normalise],
               "pv_pu = pv_kw / (installed capacity); nc_pu = nc_kw / (maximum export).")
ex2 = Exercise("ex2", "Ex2 linear schedule", _check2, [ref.linear_schedule],
               "torch.linspace(beta_1, beta_K, K) gives β_1..β_K; put a 0 in front with torch.cat; "
               "α = 1 − β; ᾱ = torch.cumprod(α, dim=0).")
ex3 = Exercise("ex3", "Ex3 closed-form x_k", _check3, [ref.q_sample],
               "a = abar[k] has shape [B]; a.unsqueeze(-1) makes it [B, 1]; then √a·x0 + √(1 − a)·eps.",
               errors=[_ERR_BROADCAST])
ex4a = Exercise("ex4a", "Ex4a shapes through the U-Net", _check4a, ["SHAPES"],
                "Channels: 4 → 128 → 256 → 512 → (1024) → 512 → 256 → 128 → 1. Lengths: 48 → 24 → 12 → 24 → 48.")
ex4b = Exercise("ex4b", "Ex4b complete the U-Net", _check4b, [model.UNet1D],
                "FiLM: scale, shift = self.film(t_emb).chunk(2, dim=-1); h * (1 + scale.unsqueeze(-1)) + "
                "shift.unsqueeze(-1). Skip: torch.cat([x, skip], dim=1).",
                errors=[("Sizes of tensors must match", "concatenate along the channel axis: torch.cat([x, skip], dim=1)."),
                        ("must match the size of tensor", "scale and shift are [B, C]: add a time axis with .unsqueeze(-1).")],
                show=[model.ModulatedConv1D, model.UNet1D])
ex4c = Exercise("ex4c", "Ex4c loss of one training iteration", _check4c, [ref.diffusion_loss],
                "k = torch.randint(1, K + 1, (B,)); eps = torch.randn_like(x0); x_k = q_sample(x0, k, eps, abar); "
                "loss = F.mse_loss(model(x_k, k, c), eps).", errors=[_ERR_BROADCAST])
ex5 = Exercise("ex5", "Ex5 one reverse step", _check5, [ref.reverse_step],
               "mean = (x_k − β_k/√(1 − ᾱ_k)·ε̂)/√α_k; add √β_k·z only if k > 1.")
ex5b = Exercise("ex5b", "Ex5b model's guess of x_0", _check5b, [ref.predict_x0],
                "Solve x_k = √ᾱ_k·x_0 + √(1 − ᾱ_k)·ε for x_0, with ε replaced by ε̂.")
ex6 = Exercise("ex6", "Ex6 post-processing", _check6, [ref.postprocess],
               "samples.clamp(min=0) and multiply by the boolean mask (ghi > 0), which broadcasts over the samples.")
ex7 = Exercise("ex7", "Ex7 median, PI band and RMSE", _check7, [ref.point_and_band, ref.rmse],
               "torch.quantile(samples, torch.tensor([0.5, 0.5 - level/200, 0.5 + level/200]), dim=0); "
               "RMSE = √(mean of squared errors).", errors=[_ERR_QUANTILE])
ex8 = Exercise("ex8", "Ex8 PICP and PINAW", _check8, [ref.picp_pinaw],
               "sun = ghi > 0; inside = (pv >= lower) & (pv <= upper); PICP = 100·mean(inside[sun]); "
               "PINAW = mean((upper − lower)[sun]) / capacity.")
ex9 = Exercise("ex9", "Ex9 pinball loss", _check9, [ref.pinball],
               "err = pv − q_pred ([Q, T]); loss = max(q·err, (q − 1)·err) with q[:, None]; mean over daylight columns.")

# ---------------------------------------------------------------- the questions
q1 = Question("q1", "C", {
    "A": "An export limit curtails PV only when the house cannot use it. Behind the meter, PV still serves the load and "
         "can produce far more than 0.3 kW, so x_0 is not capped.",
    "B": "Shading or degradation lowers PV in pu of capacity; normalising does not remove it. And it would not explain "
         "why the export in particular stays so small.",
    "C": "A small maximum export means the PV rarely leaves the house: imports dominate NC and become large numbers in "
         "pu (hence the 25 pu above). The midday export, the strongest PV signal in NC_d, is weak or absent, so the "
         "model must lean more on GHI and NC_d−1. You meet such households again in Section 6.",
    "D": "Only the export side is bounded (NC ≥ −1 pu). Imports are divided by the same small number and become large: "
         "this household produces the largest pu values in the dataset."})
q2 = Question("q2", "A", {
    "A": "This day's PV varies by about 0.33 pu, while the noise has standard deviation 1. At k = 10 the noise share "
         "is 0.30, so the added noise (0.30) is already as large as the remaining signal variation (0.95 × 0.33). "
         "The shape drowns long before the shares cross at k = 27, as C3-A5 noted: PV values are small compared "
         "with the noise.",
    "B": "The opposite: β_k grows with k, from 10⁻⁴ to 0.2, so the first steps add the least noise. ᾱ_10 is still "
         "about 0.91.",
    "C": "The noise is indeed independent per half-hour, but that holds at every k and for any signal. It does not "
         "explain why the shape is gone while 95 % of the signal amplitude remains.",
    "D": "√ᾱ_10 ≈ 0.95: the profile keeps 95 % of its size. It is not shrunk away, it is drowned by noise of a similar "
         "size."})
q3 = Question("q3", "D", {
    "A": "At the bottleneck one step is 4 half-hours, so each kernel-3 convolution adds 2 × 4 = 8: 28 → 36 → 44. "
         "Close, but still short of 48.",
    "B": "Channels add capacity at each position, not reach: the receptive field depends only on kernels and poolings.",
    "C": "A kernel-2, stride-2 convolution reaches exactly as far as a 2-wide pooling: the receptive field stays 28.",
    "D": "Each extra level adds a pooling, which doubles the reach of every later kernel: 3 → 5 → 6 → 10 → 12 → 20 → 24 "
         "→ 40 → 56 half-hours ≥ 48. The price: the bottleneck holds only 6 values and the network has more weights. "
         "Since 48 = 2⁴·3, at most 4 levels are possible."})
q4 = Question("q4", "B", {
    "A": "k ≈ 27 is where x_k itself becomes signal-dominated. x̂_0 subtracts the predicted noise, so it can show the "
         "shape much earlier. Run the next cells and compare the two rows of correlations.",
    "B": "Already at k ≈ 80 the median correlation of x̂_0 with the true PV is about 0.6 on the sunny days, and above "
         "0.9 by k = 60, while x_k itself only shows the shape around k = 10. The network guesses the shape early; "
         "the later steps refine detail. On an overcast day the shape appears much later: there is little PV to "
         "recover (day c: median correlation of x̂_0 about 0.1 at k = 80, 0.5 at k = 50, 0.8 at k = 10).",
    "C": "The amplification is real: 1/√ᾱ_80 ≈ 29. But ε̂ is accurate enough that x̂_0 at k = 80 is noisy, not "
         "shapeless. It only fails at k = 100, where the factor is about 216.",
    "D": "x̂_0 uses the network's estimate of ε as well as x_k, so it can be far better than x_k. Run the next cells and "
         "compare the two rows of correlations."})
q5 = Question("q5", "D", {
    "A": "Physically yes, but the network knows no physics: it learned which input explained PV best in its training "
         "data. Run the cell and see.",
    "B": "A reasonable guess, but no: the samples do not land in between. In (a) the PV even went down although the sky "
         "was clear; in (b) it nearly vanished.",
    "C": "In (a) the sky signalled much more PV, yet the samples went down: the meter wins in both directions.",
    "D": "The samples follow the meter. In the 2012–13 Ausgrid data, a clear day almost always means a midday export, "
         "so the model learned 'no export ⇒ no PV'. It learned from data, not physics."})
q6 = Question("q6", "C", optional=True, feedback={
    "A": "1 % of 100 samples is a single sample.",
    "B": "Batching runs the samples in parallel, but each one still needs K = 100 passes: 100 sequential steps, "
         "50 000 sample-passes of computation.",
    "C": "1 % of 500 is 5, and each sample costs K = 100 network passes. Tails are expensive: that is why the 10 % and "
         "90 % quantiles in the plot settle more slowly than the median, and why wide intervals need many samples "
         "(C3-A6).",
    "D": "The √N law describes how fast the estimate improves, but the question asks for 5 samples below the 1 % "
         "quantile: 1 % of 5 000 is 50, ten times more than needed."})
q7 = Question("q7", "A", {
    "A": "The cause is the training data: in the 2012–13 Ausgrid data almost no household absorbed its PV, so 'no export "
         "⇒ no PV' held. Data with batteries and flexible daytime loads teaches the model otherwise. This is a "
         "distribution shift, not a flaw of the method.",
    "B": "More samples estimate the same, wrong distribution more precisely: you get a tighter interval around a wrong "
         "answer.",
    "C": "NC_d is the most informative condition on sunny days. Without it you get the GHI-only model, the least "
         "accurate one in the comparison below.",
    "D": "A household with a battery shows no export on earlier days either: the longer history repeats the same "
         "evidence, and the model has still never seen 'PV without export'."})
q8 = Question("q8", "B", {
    "A": "The pinball loss rewards quantiles in the right place, not width: an interval wider than needed also costs "
         "loss at every quantile.",
    "B": "PINAW only measures width. The pinball loss penalises every quantile that is in the wrong place, so tight but "
         "misplaced intervals (NC only on dull days, when low PV leaves NC ambiguous) score worse. A few days are an "
         "illustration, not an evaluation: without a GPU (3 days, 20 samples) the two models come very close.",
    "C": "Your pinball function (Ex9) uses the daylight half-hours only, as PICP and PINAW do.",
    "D": "Both columns are means over the days (see the line printed above the table): they are comparable."})

EXERCISES = [ex1, ex2, ex3, q1, q2, ex4a, ex4b, q3, ex4c, ex5, ex5b, q4, ex6, ex7, ex8, q6, ex9, q5, q7, q8]


def progress():
    """✅/⬜ per exercise and question (kept across runtime restarts)."""
    s = _load_state()
    done = sum(s.get(e.name, False) for e in EXERCISES if not e.optional)
    print(f"Progress: {done}/{sum(not e.optional for e in EXERCISES)} (optional items not counted)")
    for e in EXERCISES:
        print(f"  {'✅' if s.get(e.name) else '⬜'} {e.title}{' (optional)' if e.optional else ''}")


def catch_up(section: int):
    """Rebuild the state a section needs (reference solutions, course checkpoints), e.g. after a runtime restart."""
    S = types.SimpleNamespace(day=data.day(data.CUSTOMER, data.DAYS_134["b"]))
    if section >= 2:
        S.beta, S.alpha, S.abar = ref.linear_schedule(model.K)
        S.schedule = (S.beta, S.alpha, S.abar)
    if section >= 4:
        S.model = model.load_pretrained("all")
    if section == 5:  # samples of the course model for days a, b, c (section 5 only)
        S.days = {label: data.day(data.CUSTOMER, d) for label, d in data.DAYS_134.items()}
        c = torch.stack([d["c"] for d in S.days.values()])
        raw = model.sample(S.model, c, S.schedule, ref.reverse_step)
        S.samples = {label: ref.postprocess(raw[i], d["ghi"]) for i, (label, d) in enumerate(S.days.items())}
    if section >= 6:
        S.models = {name: model.load_pretrained(name) for name in model.CONDITIONS}
    print(f"✓ state for section {section} ready: {', '.join(vars(S))}")
    return S
