"""Hosted Ausgrid subset (see DATA_README.md): download, normalised tensors, day selection."""
import urllib.request
from functools import lru_cache

import numpy as np
import torch

from . import ASSET_URL, CACHE

T = 48
TIME = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]


def fetch(name: str):
    """Path of a release asset, downloaded once into the cache."""
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(ASSET_URL + name, path.with_suffix(".part"))
        path.with_suffix(".part").rename(path)
    return path


@lru_cache(maxsize=None)
def load() -> dict:
    with np.load(fetch("ausgrid_c3.npz")) as f:
        return {k: f[k] for k in f.files}


def _iso(yyyymmdd) -> str:
    s = str(int(yyyymmdd))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def experiment_days() -> list:
    """Section 6 days: customer 134 (days a, b, c), then 5 calibration customer-days across the seasons."""
    return [(int(c), _iso(d)) for c, d in load()["experiment_days"]]


CUSTOMER = 134  # the household of C3-A5 and C3-A6
DAYS_134 = {"a": "2012-12-13", "b": "2013-01-04", "c": "2013-01-28"}  # drop, clear, overcast


def conditions(rows) -> torch.Tensor:
    """Conditions c = (GHI_d, NC_d, NC_d-1) of the given rows, shape [len(rows), 3, T]."""
    d = load()
    return torch.from_numpy(np.stack([d["ghi"][rows], d["nc"][rows], d["nc_lag1"][rows]], 1))


def train_set():
    """x_0 [N, T] (PV / capacity) and c [N, 3, T] of the 107 training customers."""
    d = load()
    rows = np.isin(d["customer"], d["customers"][d["is_train"]])
    return torch.from_numpy(d["pv"][rows]), conditions(rows)


def day(customer: int, date: str) -> dict:
    """One day of one customer. Values in pu (model inputs) and in kW (for people)."""
    d = load()
    row = np.flatnonzero((d["customer"] == customer) & (d["date"] == int(date.replace("-", ""))))
    assert len(row) == 1, f"no data for customer {customer} on {date}"
    i = int(np.flatnonzero(d["customers"] == customer)[0])
    cap, mexp = float(d["capacity_kwp"][i]), float(d["max_export_kw"][i])
    c = conditions(row)[0]
    pv = torch.from_numpy(d["pv"][row[0]])
    return {"customer": customer, "date": date, "capacity_kwp": cap, "max_export_kw": mexp,
            "pv_pu": pv, "c": c, "ghi": c[0], "nc_pu": c[1], "nc_lag1_pu": c[2],
            "pv_kw": pv * cap, "nc_kw": c[1] * mexp, "nc_lag1_kw": c[2] * mexp}

