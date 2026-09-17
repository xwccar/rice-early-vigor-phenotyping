from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture

from release_utils import load_yaml, resolve_repo_path


def hartigan_dip_test(values: np.ndarray, seed: int = 41, bootstrap: int = 1000) -> dict:
    """Hartigan's dip test using the small, compiled ``diptest`` package."""
    try:
        from diptest import diptest
    except ImportError as exc:
        raise RuntimeError("Hartigan's dip test requires `pip install diptest`.") from exc
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) < 4:
        raise ValueError("Hartigan's dip test requires at least four finite observations")
    statistic, p_value = diptest(x, boot_pval=True, n_boot=int(bootstrap), seed=int(seed))
    return {"n": len(x), "dip": float(statistic), "bootstrap_p_value": float(p_value)}


def _kde_mode_count(values: np.ndarray, bandwidth: float, grid_size: int = 1024) -> int:
    values = np.asarray(values, dtype=float)
    scale = max(float(np.std(values, ddof=1)), 1e-3)
    margin = 4.0 * max(bandwidth, scale)
    limits = (float(values.min() - margin), float(values.max() + margin))
    histogram, edges = np.histogram(values, bins=grid_size, range=limits, density=True)
    step = float(edges[1] - edges[0])
    density = gaussian_filter1d(histogram.astype(float), sigma=max(bandwidth / step, 0.35), mode="constant")
    derivative = np.diff(density)
    # Ignore numerical shoulders with negligible density.
    peaks = (derivative[:-1] > 0) & (derivative[1:] <= 0)
    peak_height = density[1:-1]
    return int(np.count_nonzero(peaks & (peak_height > density.max() * 1e-7)))


def _critical_bandwidth(values: np.ndarray, modes: int = 1) -> float:
    values = np.asarray(values, dtype=float)
    scale = float(np.std(values, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Silverman's test requires nonconstant finite values")
    lower, upper = scale * 1e-4, scale
    while _kde_mode_count(values, upper) > modes:
        upper *= 2.0
    for _ in range(32):
        middle = 0.5 * (lower + upper)
        if _kde_mode_count(values, middle) <= modes:
            upper = middle
        else:
            lower = middle
    return float(upper)


def silverman_unimodality_test(values: np.ndarray, seed: int = 41, bootstrap: int = 1000) -> dict:
    """Smoothed-bootstrap Silverman test of one mode versus more than one."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        raise ValueError("Silverman's test requires at least four finite observations")
    x = (x - x.mean()) / x.std(ddof=1)
    observed = _critical_bandwidth(x, modes=1)
    rng = np.random.default_rng(seed)
    critical = np.empty(int(bootstrap), dtype=float)
    denominator = np.sqrt(1.0 + observed**2)
    for index in range(int(bootstrap)):
        sample = rng.choice(x, size=len(x), replace=True)
        sample = (sample + rng.normal(0.0, observed, size=len(x))) / denominator
        critical[index] = _critical_bandwidth(sample, modes=1)
    p_value = (1 + np.count_nonzero(critical >= observed)) / (len(critical) + 1)
    return {
        "n": len(x),
        "critical_bandwidth": observed,
        "bootstrap_replicates": int(bootstrap),
        "bootstrap_p_value": float(p_value),
    }


def holm_adjust(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def normalize_results(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {"R2": "r2", "RMSE": "rmse", "RRMSE": "rrmse_percent", "ME": "me_pred_minus_true", "value": "metric_value"}
    frame = frame.rename(columns=rename)
    if {"metric", "metric_value"}.issubset(frame.columns):
        identifiers = [column for column in frame.columns if column not in {"metric", "metric_value"}]
        frame = frame.pivot_table(index=identifiers, columns="metric", values="metric_value", aggfunc="first").reset_index()
    if "model" not in frame and "control" in frame:
        frame["model"] = frame["control"]
    if "experiment_id" not in frame:
        frame["experiment_id"] = "combined_results"
    if "statistical_family" not in frame:
        mapping = {
            "pointcloud_dat_late_fusion_fp32_legacy_protocol": "pointcloud_backbones",
            "multimodal_baselines_5seeds": "temporal_and_fusion_strategies",
            "multimodal_ogr_uniform_5seeds": "temporal_and_fusion_strategies",
            "target_reconstruction_controls_eq2": "target_reconstruction_controls",
        }
        frame["statistical_family"] = frame["experiment_id"].map(mapping).fillna(frame["experiment_id"])
    if "target" not in frame:
        frame["target"] = "vigor_score"
    return frame


def repeated_model_tests(results: pd.DataFrame, metrics: list[str], alpha: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus, pairwise = [], []
    for (family, target), group in results.groupby(["statistical_family", "target"], dropna=False):
        for metric in metrics:
            if metric not in group:
                continue
            if group.duplicated(["seed", "model"]).any():
                raise ValueError(f"Duplicate seed/model results in {family}/{target}")
            pivot = group.pivot(index="seed", columns="model", values=metric)
            if pivot.isna().any().any() or not np.isfinite(pivot.to_numpy()).all():
                raise ValueError(f"Incomplete or nonfinite paired results in {family}/{target}")
            if pivot.shape[0] < 3 or pivot.shape[1] < 2:
                continue
            p_value = np.nan
            if pivot.shape[1] >= 3:
                statistic, p_value = stats.friedmanchisquare(*(pivot[column] for column in pivot))
                omnibus.append({"statistical_family": family, "target": target, "metric": metric, "n_seeds": len(pivot), "n_models": pivot.shape[1], "friedman_chi2": statistic, "p_value": p_value})
            if pivot.shape[1] < 3 or not np.isfinite(p_value) or p_value >= alpha:
                continue
            local = []
            for first, second in itertools.combinations(pivot.columns, 2):
                difference = pivot[first] - pivot[second]
                try:
                    statistic, p_value = stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
                except ValueError:
                    statistic, p_value = 0.0, 1.0
                local.append({"statistical_family": family, "target": target, "metric": metric, "model_1": first, "model_2": second, "n_seeds": len(difference), "mean_difference_model1_minus_model2": difference.mean(), "wilcoxon_statistic": statistic, "p_value": p_value})
            adjusted = holm_adjust([row["p_value"] for row in local])
            for row, value in zip(local, adjusted):
                row["p_value_holm"] = value
            pairwise.extend(local)
    return pd.DataFrame(omnibus), pd.DataFrame(pairwise)


def gaussian_mixture_tests(values: np.ndarray, seed: int = 41, bootstrap: int = 1000) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)].reshape(-1, 1)
    one = GaussianMixture(1, random_state=seed, n_init=20).fit(x)
    two = GaussianMixture(2, random_state=seed, n_init=20).fit(x)
    observed_lr = 2 * (two.score(x) - one.score(x)) * len(x)
    rng = np.random.default_rng(seed)
    bootstrap_lr = []
    for index in range(bootstrap):
        sample = rng.normal(float(one.means_[0, 0]), float(np.sqrt(one.covariances_[0, 0, 0])), size=len(x)).reshape(-1, 1)
        m1 = GaussianMixture(1, random_state=seed + index + 1, n_init=3).fit(sample)
        m2 = GaussianMixture(2, random_state=seed + index + 1, n_init=3).fit(sample)
        bootstrap_lr.append(2 * (m2.score(sample) - m1.score(sample)) * len(sample))
    order = np.argsort(two.means_.ravel())
    return {
        "n": len(x), "bic_one_component": one.bic(x), "bic_two_components": two.bic(x),
        "delta_bic_one_minus_two": one.bic(x) - two.bic(x), "likelihood_ratio": observed_lr,
        "bootstrap_p_value": (1 + np.sum(np.asarray(bootstrap_lr) >= observed_lr)) / (bootstrap + 1),
        "component_means": two.means_.ravel()[order].tolist(), "component_weights": two.weights_[order].tolist(),
    }


def hc3_linear_regression(x: np.ndarray, y: np.ndarray) -> dict:
    design = np.column_stack((np.ones(len(x)), x))
    inverse = np.linalg.pinv(design.T @ design)
    beta = inverse @ design.T @ y
    residual = y - design @ beta
    leverage = np.sum((design @ inverse) * design, axis=1)
    meat = design.T @ np.diag((residual / np.clip(1 - leverage, 1e-12, None)) ** 2) @ design
    covariance = inverse @ meat @ inverse
    standard_error = np.sqrt(np.diag(covariance))
    t_value = beta / standard_error
    p_value = 2 * stats.t.sf(np.abs(t_value), df=max(len(x) - 2, 1))
    return {"intercept": beta[0], "slope": beta[1], "slope_hc3_se": standard_error[1], "slope_hc3_p": p_value[1], "r2": 1 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)}


def vigor_group_yield_tests(reference: pd.DataFrame, score_column: str = "vigor_score", bootstrap: int = 1000) -> tuple[dict, pd.DataFrame]:
    if score_column not in reference:
        raise ValueError(f"Configured score column is missing: {score_column}")
    output = {"score_column": score_column}
    values = pd.to_numeric(reference[score_column], errors="coerce").dropna().to_numpy()
    output["hartigan_dip"] = hartigan_dip_test(values, bootstrap=bootstrap)
    output["silverman_unimodality"] = silverman_unimodality_test(values, bootstrap=bootstrap)
    output["distribution"] = gaussian_mixture_tests(values, bootstrap=bootstrap)
    if {"group", score_column}.issubset(reference.columns):
        groups = {name: pd.to_numeric(part[score_column], errors="coerce").dropna().to_numpy() for name, part in reference.groupby("group")}
        if len(groups) == 2:
            names = list(groups)
            statistic, p = stats.mannwhitneyu(groups[names[0]], groups[names[1]], alternative="two-sided")
            output["group_comparison"] = {"group_1": names[0], "group_2": names[1], "median_1": np.median(groups[names[0]]), "median_2": np.median(groups[names[1]]), "mann_whitney_u": statistic, "p_value": p}
    paired = reference[[score_column, "yield_g_per_plant"]].apply(pd.to_numeric, errors="coerce").dropna() if "yield_g_per_plant" in reference else pd.DataFrame()
    quadrants = pd.DataFrame()
    if len(paired) >= 3:
        x, y = paired[score_column].to_numpy(), paired["yield_g_per_plant"].to_numpy()
        pearson, pearson_p = stats.pearsonr(x, y)
        spearman, spearman_p = stats.spearmanr(x, y)
        output["yield_correlation"] = {"n": len(x), "pearson_r": pearson, "pearson_p": pearson_p, "spearman_rho": spearman, "spearman_p": spearman_p, **hc3_linear_regression(x, y)}
        score_low, score_high = np.quantile(x, [0.25, 0.75])
        yield_low, yield_high = np.quantile(y, [0.25, 0.75])
        paired = paired.assign(score_class=np.where(x <= score_low, "low", np.where(x >= score_high, "high", "middle")), yield_class=np.where(y <= yield_low, "low", np.where(y >= yield_high, "high", "middle")))
        quadrants = paired.query("score_class != 'middle' and yield_class != 'middle'").groupby(["score_class", "yield_class"]).size().rename("count").reset_index()
        quadrants["percent_of_all_paired"] = 100 * quadrants["count"] / len(paired)
        output["quartile_thresholds"] = {"score_q25": score_low, "score_q75": score_high, "yield_q25": yield_low, "yield_q75": yield_high}
    return output, quadrants


def parse_args():
    parser = argparse.ArgumentParser(description="Run manuscript statistical analyses")
    parser.add_argument("--config", default="config/statistics.yaml")
    parser.add_argument("--results", default=None, help="Optional complete rerun result table; overrides config.results")
    parser.add_argument("--score-column", default=None, help="Explicit reference-score column; overrides config.score_column")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=None,
        help="Bootstrap replicates; overrides config.bootstrap_replicates",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)
    output = resolve_repo_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    results_path = resolve_repo_path(args.results or config["results"])
    results = normalize_results(pd.read_csv(results_path))
    omnibus, pairwise = repeated_model_tests(results, list(config["metrics"]), float(config.get("alpha", 0.05)))
    summary = (
        results.groupby(["statistical_family", "target", "model"], dropna=False)[list(config["metrics"])]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary.to_csv(output / "descriptive_summary.csv", index=False)
    omnibus.to_csv(output / "friedman_tests.csv", index=False)
    pairwise.to_csv(output / "wilcoxon_holm_tests.csv", index=False)
    reference = pd.read_csv(resolve_repo_path(config["reference_vigor"]))
    score_column = args.score_column or config.get("score_column", "vigor_score")
    bootstrap = int(args.bootstrap if args.bootstrap is not None else config.get("bootstrap_replicates", 1000))
    study, quadrants = vigor_group_yield_tests(reference, score_column=score_column, bootstrap=bootstrap)
    (output / "vigor_group_yield_statistics.json").write_text(json.dumps(study, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    quadrants.to_csv(output / "yield_quartile_quadrants.csv", index=False)
    print(f"Saved statistical outputs to {output}")


if __name__ == "__main__":
    main()
