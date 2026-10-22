# Data and checkpoints (release assets)

The release assets of this repository are licensed under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
The code is MIT-licensed (see `LICENSE`).

| Asset | Size | Content |
|---|---|---|
| `ausgrid_c3.npz` | 14 MB | Preprocessed subset of the Ausgrid Solar Home Electricity dataset, with irradiance |
| `pv_all.pt`, `pv_ghi.pt`, `pv_nc.pt` | 21 MB each | Course cDDPM checkpoints (PyTorch `state_dict`, float32) |
| `SHA256SUMS` | | Checksums of the four files |

## Sources and attribution

- **Ausgrid Solar Home Electricity dataset, 1 July 2012 – 30 June 2013** (Ausgrid), Creative Commons Attribution
  licence: https://data.nsw.gov.au/data/dataset/5ab48b70-5e99-47d3-9193-5c34a2676d93.
  Described in E. L. Ratnam, S. R. Weller, C. M. Kellett and A. T. Murray, "Residential load and rooftop PV
  generation: an Australian distribution network dataset," *Int. J. Sustain. Energy*, vol. 36, no. 8,
  pp. 787–806, 2017.
- **Irradiance (GHI):** Open-Meteo historical weather API, `shortwave_radiation`, CC BY 4.0. P. Zippenfenig,
  "Open-Meteo.com Weather API," Zenodo, 2023, doi: 10.5281/zenodo.7970649.

Ausgrid has taken its download page down, so the course hosts the subset it needs here.

## What `ausgrid_c3.npz` contains

These are real values with no added noise, at a 30-minute resolution: 48 values per day, starting at 00:00. The days
run from 2 July 2012 to 30 June 2013 (364 days per customer). 1 July is dropped because NC_d−1 does not exist for it.

| Array | Shape | Meaning |
|---|---|---|
| `pv` | [46 592, 48] | PV generation ÷ installed PV capacity (pu) |
| `nc` | [46 592, 48] | Net consumption NC_d = consumption − PV, positive = import, ÷ the customer's maximum export (pu) |
| `nc_lag1` | [46 592, 48] | Same, previous day (NC_d−1) |
| `ghi` | [46 592, 48] | GHI_d in kW/m² (Open-Meteo, W/m² ÷ 1000) |
| `customer`, `date` | [46 592] | Ausgrid customer ID and date (YYYYMMDD) of each row |
| `customers` | [128] | Customer IDs |
| `capacity_kwp` | [128] | Installed PV capacity from the Ausgrid metadata (kWp) |
| `max_export_kw` | [128] | Maximum export seen by the meter over the year (kW) |
| `is_train` | [128] | Training customer of the course models (107 customers) |
| `experiment_days` | [8, 2] | (customer, YYYYMMDD) of the days used in section 6 of the notebook |

- **Customers.** The subset holds the 107 training customers of the course models, test customer 134 (used in
  C3-A5 to C3-A7) and the 20 test customers of the C3-A7 calibration: 128 of the 300 customers. The train/test split is
  the course split (43 irregular customers dropped, then a random split by customer with seed 0).
- **Identifiers.** The Ausgrid customer IDs are kept; they are anonymous in the original dataset. Postcodes are not
  included.
- **GHI.** It was retrieved at each customer's postcode centroid, hourly, shifted by −30 min and linearly
  interpolated to 30 min. Daylight-saving gaps are filled with 0.
- **kW.** To get kW back, multiply `pv` by `capacity_kwp` and `nc` by `max_export_kw`.

## Quirks worth knowing

- **PV up to 1.65 pu.** Some systems produce more than their listed capacity. Typical reasons are a nameplate
  capacity that understates the array, metadata errors or metering effects. The values were left as they are.
- **NC up to 25 pu.** NC is divided by the customer's maximum export. A household that rarely exports has a small
  maximum export, so its imports become large numbers in pu.

## Checkpoints

The checkpoints are the course models of C3-A5 to C3-A7. They were trained on the 107 training customers of this subset
only:

- **Architecture and training:** 1D U-Net of depth 2, 128 base channels, sinusoidal time embedding of size 32 with
  FiLM, ε-prediction, 2000 gradient steps, batch 256, Adam with lr 1e-3, seed 0.
- **Course noise schedule:** linear β from 1e-4 to **0.2**, K = 100. The paper uses 0.02.
- **Conditions:** `pv_all` uses GHI_d, NC_d and NC_d−1; `pv_ghi` uses GHI_d; `pv_nc` uses NC_d and NC_d−1.
- **Code:** paper code commit `1dd99af`.
- **Status:** they are course illustrations, not the models of the paper.

Please cite the sources above and M. Jené-Vinuesa et al., "Conditional diffusion modeling for probabilistic
behind-the-meter PV disaggregation," *Energy and AI*, 2026, doi: 10.1016/j.egyai.2026.100692.
