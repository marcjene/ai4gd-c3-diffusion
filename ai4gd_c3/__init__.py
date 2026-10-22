"""AI4GreenDeal course C3, asset A7: hands-on conditional diffusion (cDDPM) for PV disaggregation."""
import os
from pathlib import Path

import torch

# The only place in the package that names the public repository. After a transfer, change this line,
# the Colab badge and the install cell of the notebook, and publish release RELEASE in the new repository.
REPO = "marcjene/ai4gd-c3-diffusion"
RELEASE = "v1"
ASSET_URL = f"https://github.com/{REPO}/releases/download/{RELEASE}/"
CACHE = Path(os.environ.get("AI4GD_C3_CACHE", Path.home() / ".cache" / "ai4gd_c3"))
DEVICE = os.environ.get("AI4GD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

from . import data, model, reference, exercises, plots  # noqa: E402,F401
from .exercises import progress, catch_up  # noqa: E402,F401
