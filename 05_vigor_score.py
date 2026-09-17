"""Plant-level construction of the phenology-derived vigor proxy in Eq. (2)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def first_attainment_time(values, threshold, times, no_attainment_time=40.0):
    crossing = np.flatnonzero(np.asarray(values) >= threshold)
    if not len(crossing):
        return float(no_attainment_time), False
    i = int(crossing[0])
    if i == 0:
        return float(times[0]), True
    fraction = (threshold - values[i-1]) / (values[i] - values[i-1])
    return float(times[i-1] + fraction * (times[i] - times[i-1])), True


def derive_leaf_rate(leaf_age, times):
    return np.gradient(np.asarray(leaf_age, float), np.asarray(times, float), edge_order=2)


def derive_leaf_acceleration(leaf_age, times):
    return np.gradient(derive_leaf_rate(leaf_age, times), np.asarray(times, float), edge_order=2)


def build_reference(frame, no_attainment_time=40.0):
    rows = []
    for uid, part in frame.groupby("plant_uid", sort=True):
        part = part.sort_values("DAT")
        if len(part) != 10 or part.DAT.duplicated().any():
            raise ValueError(f"Expected ten unique dates for {uid}")
        if part.maturity_productive_tiller.nunique() != 1:
            raise ValueError(f"Inconsistent plant-specific maturity threshold for {uid}")
        t = part.DAT.to_numpy(float)
        a, attained = first_attainment_time(part.manual_tiller.to_numpy(float), float(part.maturity_productive_tiller.iloc[0]), t, no_attainment_time)
        acceleration = derive_leaf_acceleration(part.manual_leaf_age.to_numpy(float), t)
        b = float(t[np.argmax(acceleration)])
        row = part.iloc[0]
        rows.append({"plant_uid": uid, "plant_id": row.plant_id, "variety_id": row.variety_id,
                     "plot_id": row.plot_id, "source_area": row.source_area, "group": row.group,
                     "a_days": a, "b_days": b, "ax": a, "bx": b,
                     "maturity_productive_tiller": int(row.maturity_productive_tiller),
                     "threshold_attained": attained, "yield_g_per_plant": row.yield_g_per_plant})
    result = pd.DataFrame(rows)
    for col, output in [("a_days", "normalized_a"), ("b_days", "normalized_b")]:
        lo, hi = result[col].min(), result[col].max()
        if hi <= lo:
            raise ValueError(f"Degenerate timing range: {col}")
        result[output] = (result[col] - lo) / (hi - lo)
    result["vigor_score"] = 1 - (result.normalized_a + result.normalized_b) / 2
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/manual_traits.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/reference_vigor.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_reference(pd.read_csv(args.input)).to_csv(args.output, index=False)
