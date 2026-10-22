# AI4GreenDeal C3-A7: conditional diffusion for PV disaggregation (hands-on)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcjene/ai4gd-c3-diffusion/blob/main/AI4GD_C3-A7.ipynb)

A Colab notebook and a small Python package for asset A7 of the AI4GreenDeal course C3, *Conditional diffusion
modeling for energy disaggregation*. Learners build, train and evaluate a conditional denoising diffusion model
(cDDPM). The model recovers rooftop PV generation from smart-meter net consumption and irradiance.

- **Notebook:** `AI4GD_C3-A7.ipynb`. Open it in Google Colab with the badge above; it needs about 120 min on a free
  T4 GPU.
- **Package `ai4gd_c3`:** data loading (`data`), the course U-Net with training and sampling loops (`model`),
  reference solutions (`reference`), checks, hints and questions (`exercises`), and plots (`plots`).
- **Data and checkpoints:** release assets, downloaded and cached on first use. See `DATA_README.md` for sources,
  licence (CC BY 4.0) and contents.

The results are course illustrations, not the results of the paper. The course uses a different noise schedule
(β up to 0.2 instead of 0.02) and evaluates on a few days only.

Outside Colab: `pip install git+https://github.com/marcjene/ai4gd-c3-diffusion`, then open the notebook.
Tests (no pytest needed): `python tests/test_package.py` from the repository root.

Code: MIT licence. Based on M. Jené-Vinuesa et al., "Conditional diffusion modeling for probabilistic
behind-the-meter PV disaggregation," *Energy and AI*, 2026, doi: 10.1016/j.egyai.2026.100692.

## Funding

*[Placeholder: AI4GreenDeal / EU acknowledgement text and logo, to be provided.]*
