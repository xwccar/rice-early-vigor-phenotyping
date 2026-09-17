from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

from release_utils import sha256_file


REQUIRED_COLUMNS = {"sample_id", "source_las", "date", "DAT", "source_area"}


def stable_rng(seed: int, sample_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{sample_id}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def sample_and_center(path: Path, sample_id: str, n_points: int, seed: int) -> np.ndarray:
    cloud = laspy.read(path)
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64, copy=False)
    if xyz.shape[0] == 0:
        raise ValueError(f"Point cloud contains no points: {path}")
    rng = stable_rng(seed, sample_id)
    replace = xyz.shape[0] < n_points
    indices = rng.choice(xyz.shape[0], size=n_points, replace=replace)
    sampled = xyz[indices]
    sampled -= sampled.mean(axis=0, keepdims=True)
    return sampled.astype(np.float32)


def build_archives(manifest: pd.DataFrame, output_dir: Path, n_points: int, seed: int) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Manifest identifiers, paths, dates, DAT values, and source areas must not be missing")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Every manifest sample_id must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    index_records = []
    for (area, date), group in manifest.groupby(["source_area", "date"], sort=True):
        group = group.sort_values("sample_id", kind="stable").reset_index(drop=True)
        dat_values = pd.to_numeric(group["DAT"], errors="raise")
        if dat_values.nunique() != 1:
            raise ValueError(f"A source-area/date group must map to one DAT value: {area}/{date}")
        arrays = []
        for row in group.itertuples(index=False):
            source = Path(row.source_las)
            if not source.is_file():
                raise FileNotFoundError(source)
            arrays.append(sample_and_center(source, str(row.sample_id), n_points, seed))
        target = output_dir / f"{area}_DAT{int(dat_values.iloc[0]):02d}_{date}.npz"
        np.savez_compressed(
            target,
            sample_id=group["sample_id"].astype(str).to_numpy(dtype="U"),
            xyz=np.stack(arrays),
            date=group["date"].astype(str).to_numpy(dtype="U"),
            DAT=pd.to_numeric(group["DAT"]).to_numpy(dtype=np.int16),
            source_area=group["source_area"].astype(str).to_numpy(dtype="U"),
        )
        records.append(
            {
                "archive": target.name,
                "sample_count": len(group),
                "n_points": n_points,
                "preprocessing_seed": seed,
                "sha256": sha256_file(target),
            }
        )
        for row in group.itertuples(index=False):
            index_records.append(
                {
                    "sample_id": row.sample_id,
                    "archive": target.name,
                    "source_area": row.source_area,
                    "date": row.date,
                    "DAT": row.DAT,
                    "n_points": n_points,
                    "preprocessing_seed": seed,
                }
            )
    audit = pd.DataFrame(records)
    audit.to_csv(output_dir / "archive_hashes.csv", index=False)
    pd.DataFrame(index_records).to_csv(output_dir / "pointcloud_index.csv", index=False)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic centered 1,024-point NPZ archives")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"sample_id": str, "date": str})
    audit = build_archives(manifest, args.output_dir, args.n_points, args.seed)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
