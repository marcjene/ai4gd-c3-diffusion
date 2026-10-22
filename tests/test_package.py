"""Reference solutions pass every check; typical mistakes get their diagnosis; the repo string is written
only in ai4gd_c3/__init__.py, the notebook's Colab badge and install cell, and the README.

Run from the repo root (pytest not needed):  python tests/test_package.py
"""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ai4gd_c3 as c3  # noqa: E402
from ai4gd_c3 import exercises as X, model, reference as ref

ROOT = Path(__file__).resolve().parents[1]


def out(ex, *args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        ex.check(*args)
    return buf.getvalue()


def test_references_pass():
    pairs = [(X.ex1, ref.normalise), (X.ex2, ref.linear_schedule), (X.ex3, ref.q_sample), (X.ex4a, ref.SHAPES),
             (X.ex4b, model.UNet1D), (X.ex4c, ref.diffusion_loss), (X.ex5, ref.reverse_step),
             (X.ex5b, ref.predict_x0), (X.ex6, ref.postprocess), (X.ex8, ref.picp_pinaw), (X.ex9, ref.pinball)]
    for ex, fn in pairs:
        assert out(ex, fn).startswith("✅"), (ex.name, out(ex, fn))
    assert out(X.ex7, ref.point_and_band, ref.rmse).startswith("✅")


def test_blanks():
    ____ = X.____
    assert "not started" in out(X.ex1, lambda pv, nc, c, m: (____, ____))
    assert "not started" in out(X.ex3, lambda x0, k, e, ab: ____ * x0)
    assert "not started" in out(X.ex4a, {k: ____ for k in ref.SHAPES})
    assert "not started" in out(X.ex4c, lambda m, x0, c, ab, K: ____)


def test_diagnoses():
    sch = lambda K, b1=1e-4, bK=0.2: (torch.linspace(b1, bK, K), 1 - torch.linspace(b1, bK, K),  # noqa: E731
                                      torch.cumprod(1 - torch.linspace(b1, bK, K), 0))
    cases = [
        (X.ex1, (lambda p, n, c, m: (p / c, n / c),), "maximum export"),
        (X.ex2, (sch,), "off by one"),
        (X.ex2, (lambda K, b1=1e-4, bK=0.2: ref.linear_schedule(K, b1, 0.02),), "paper's schedule"),
        (X.ex3, (lambda x, k, e, ab: ab[k][:, None] * x + (1 - ab[k][:, None]) * e,), "√ missing"),
        (X.ex3, (lambda x, k, e, ab: ab[k].sqrt() * x + (1 - ab[k]).sqrt() * e,), "unsqueeze"),
        (X.ex3, (lambda x, k, e, ab: ref.q_sample(x, k - 1, e, ab),), "off by one"),
        (X.ex4a, ({**ref.SHAPES, "decoder_2": (16, 512, 24)},), "decoder_2"),
        (X.ex4a, ({**ref.SHAPES, "output": (16, 1, 48)},), "squeeze"),
        (X.ex5, (lambda x, e, k, b, a, ab, z: (x - b[k] / (1 - ab[k]).sqrt() * e) / a[k].sqrt() + b[k].sqrt() * z,), "last step"),
        (X.ex5, (lambda x, e, k, b, a, ab, z: ref.reverse_step(x, e, k, b, a, ab, z * b[k].sqrt()),), "σ_k = √β_k"),
        (X.ex6, (lambda s, g: s.clamp(min=0),), "GHI = 0"),
        (X.ex8, (lambda pv, lo, up, g, cap: (100 * ((pv >= lo) & (pv <= up)).float().mean(), ((up - lo) / cap).mean()),), "daylight"),
        (X.ex8, (lambda *a: (ref.picp_pinaw(*a)[0] / 100, ref.picp_pinaw(*a)[1]),), "percentage"),
        (X.ex9, (lambda pv, qp, q, g: ref.pinball(pv, qp, 1 - q, g),), "swapped"),
        (X.ex7, (lambda s, l: torch.quantile(s, torch.tensor([0.5, 0.5 - l / 2, 0.5 + l / 2]), dim=0), ref.rmse), "fractions"),
        (X.ex7, (lambda s, l: ref.point_and_band(s, l), lambda a, b: ((a - b) ** 2).mean()), "MSE"),
    ]
    for ex, args, expect in cases:
        o = out(ex, *args)
        assert o.startswith("❌") and expect in o, (ex.name, o)


def test_loss_off_by_one():
    def bad(m, x0, c, ab, K):
        k = torch.randint(0, K, (len(x0),))
        e = torch.randn_like(x0)
        return F.mse_loss(m(ref.q_sample(x0, k, e, ab), k, c), e)
    assert "k = 0" in out(X.ex4c, bad)

    def predicts_x0(m, x0, c, ab, K):
        k = torch.randint(1, K + 1, (len(x0),))
        return F.mse_loss(m(ref.q_sample(x0, k, torch.randn_like(x0), ab), k, c), x0)
    assert "not x_0" in out(X.ex4c, predicts_x0)


def test_unet_diagnoses():
    class NoFilm(model.ModulatedConv1D):
        def forward(self, x, t_emb):
            return F.relu(self.conv(x))

    class Unet(model.UNet1D):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            for name, m in list(self.named_modules()):
                if isinstance(m, model.ModulatedConv1D):
                    m.__class__ = NoFilm
    assert "FiLM missing" in out(X.ex4b, Unet)

    class Renamed(model.UNet1D):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.last = self.final_conv
            del self.final_conv
    assert "renamed" in out(X.ex4b, Renamed)


def test_questions_and_use():
    assert out(X.q2, "b").startswith("✅") and out(X.q2, "A").startswith("❌")
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn = X.ex6.use(lambda s, g: s)
    assert fn is ref.postprocess and "reference" in buf.getvalue()


def test_repo_string_only_in_three_places():
    hits = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".ipynb", ".md", ".toml", ".txt", ".cfg"} and ".git" not in p.parts:
            text = p.read_text(errors="ignore")
            if p.name == "test_package.py":
                continue
            n = text.count(c3.REPO)
            if n:
                hits.append((p.relative_to(ROOT).as_posix(), n))
    hits = dict(hits)
    assert hits.pop("ai4gd_c3/__init__.py") == 1
    hits.pop("README.md", None)  # Colab badge and install line
    assert hits.pop("AI4GD_C3-A7.ipynb") == 2
    cells = json.loads((ROOT / "AI4GD_C3-A7.ipynb").read_text())["cells"]
    where = [i for i, c in enumerate(cells) if c3.REPO in "".join(c["source"])]
    assert len(where) == 2 and "colab-badge" in "".join(cells[where[0]]["source"]), where  # badge + install cell
    assert not hits, f"repository name written outside the allowed places: {hits}"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("passed", name)
