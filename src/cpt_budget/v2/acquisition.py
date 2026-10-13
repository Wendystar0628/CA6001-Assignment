"""Whole-profile GP information acquisition without access to hidden qc values.

The proxy kernel is frozen during selection.  Its WhiteKernel is observation
noise, not removable latent variance.  Both batch points are selected before
the replay reveals either profile.  All integrations use the public action
pool, with equal weights for its complete 24-cell profiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr
from sklearn.gaussian_process.kernels import WhiteKernel

from ..acquisition import _coordinates, _nearest, select_batch
from ..models.base import validate_query_site

POLICIES = ("gp_threshold", "gp_alc", "gp_sur")


def _settings(config):
    legacy = config.get("acquisition_v2", {})
    current = config.get("v2", {}).get("acquisition", {})
    settings = {**legacy, **current}
    settings["fantasies"] = current.get("n_fantasies", legacy.get("fantasies", 16))
    settings["alc_shortlist"] = current.get("shortlist_size", legacy.get("alc_shortlist", 8))
    if (float(settings.get("false_negative_cost", 4)) != 4
            or float(settings.get("false_positive_cost", 1)) != 1
            or float(settings.get("profile_weight", 1)) != float(settings.get("risk_weight", 1))):
        raise ValueError("The frozen v0.2 protocol requires FN:FP=4:1 and equal objective weights")
    return settings


def _validate_grid(candidate_grid, config):
    forbidden = {"qc_mpa", "log_qc", "truth_qc_mpa", "truth_log_qc", "true_low"}
    if forbidden.intersection(candidate_grid.columns):
        raise ValueError("Acquisition accepts candidate geometry only, never hidden truth")
    required = {"site_id", "location_id", "depth_index", "x_m", "y_m", "z_center_m"}
    if not required.issubset(candidate_grid):
        raise ValueError(f"Candidate grid missing {sorted(required - set(candidate_grid))}")
    grid = candidate_grid.copy()
    grid["location_id"] = grid.location_id.astype(str)
    grid = grid.sort_values(["location_id", "depth_index"], kind="stable").reset_index(drop=True)
    if grid.empty or grid.duplicated(["location_id", "depth_index"]).any():
        raise ValueError("Candidate grid needs unique nonempty location/depth cells")
    # v0.2's experiment is explicitly 2--8 m in 24 cells, not partial profiles.
    indices = tuple(range(24))
    if not grid.groupby("location_id").depth_index.apply(lambda x: tuple(x) == indices).all():
        raise ValueError("Every selectable CPT must contain all 24 depth cells")
    if not np.isfinite(grid[["x_m", "y_m", "z_center_m"]].to_numpy(float)).all():
        raise ValueError("Candidate geometry must be finite")
    return grid


def _latent_posterior(proxy, grid):
    """Return full means, latent covariance and physical observation noise.

    sklearn's return_cov includes the additive WhiteKernel diagonal.  alpha is
    only training numerical jitter in this project's GP and is not added to
    future-observation uncertainty.
    """
    validate_query_site(grid, proxy.observed)
    if proxy.regressor is None:
        raise RuntimeError("The selection proxy must already be conditioned")
    kernel = proxy.regressor.kernel_
    if not hasattr(kernel, "k2") or not isinstance(kernel.k2, WhiteKernel):
        raise ValueError("Expected the v1 GP signal + WhiteKernel observation-noise kernel")
    mean, total_cov = proxy.regressor.predict(proxy._coordinates(grid), return_cov=True)
    noise = float(kernel.k2.noise_level)
    cov = np.asarray(total_cov, dtype=float)
    cov = (cov + cov.T) * 0.5
    cov.flat[::len(cov) + 1] -= noise
    diagonal = np.diag(cov).copy()
    if diagonal.min() < -1e-7 or not np.isfinite(cov).all() or noise < 0:
        raise ValueError("Invalid latent GP posterior")
    cov.flat[::len(cov) + 1] = np.maximum(diagonal, 0.0)
    return mean + proxy._trend_at(grid), cov, noise


def _block_update(cov, indices, noise):
    """Schur update using an entire noisy observed profile, with stable solve."""
    cross = cov[:, indices]
    observation_cov = cov[np.ix_(indices, indices)].copy()
    observation_cov.flat[::len(indices) + 1] += noise
    scale = max(float(np.diag(observation_cov).max()), 1.0)
    # Numerical stabilization is tiny and is reported separately from noise.
    jitter = scale * 1e-10
    for _ in range(5):
        try:
            stabilized = observation_cov + np.eye(len(indices)) * jitter
            factor = cho_factor(stabilized, lower=True, check_finite=False)
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    else:
        raise ValueError("Candidate observation block is not positive definite")
    solve_cross = cho_solve(factor, cross.T, check_finite=False)
    reduction = np.maximum(np.einsum("ij,ji->i", cross, solve_cross), 0.0)
    after_diagonal = np.maximum(np.diag(cov) - reduction, 0.0)
    return cross, solve_cross, after_diagonal, np.linalg.cholesky(stabilized), jitter


def _probability(means, latent_variance, noise, tau, transform):
    std = np.sqrt(np.maximum(latent_variance + noise, 1e-15))
    p = ndtr((np.log(tau) - means) / std)
    if transform is not None:
        p = np.asarray(transform(p), dtype=float)
        if p.shape != means.shape:
            raise ValueError("Probability transform must preserve the input shape")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("GP risk probabilities must be finite and in [0,1]")
    return p


def _risk(means, latent_variance, noise, tau, transform, fn_cost, fp_cost):
    p = _probability(means, latent_variance, noise, tau, transform)
    return float(np.minimum(fn_cost * p, fp_cost * (1 - p)).mean())


def _stage_scores(cov, means, groups, eligible, noise, variance_scale, config,
                  probability_transform, *, policy, rng):
    """Score a marginal batch stage, averaging over preceding fantasy states."""
    settings = _settings(config)
    fantasies = int(settings.get("fantasies", 16))
    shortlist_size = int(settings.get("alc_shortlist", 8))
    tau = float(config.get("data", {}).get("tau_mpa", 1.0))
    fn_cost, fp_cost = 4.0, 1.0
    if fantasies < 1 or shortlist_size < 1 or tau <= 0:
        raise ValueError("Positive fantasy count, shortlist size, and threshold required")
    n = len(groups)
    raw_gain = np.full(n, np.nan)
    updates = {}
    for i in eligible:
        update = _block_update(cov, groups[i], noise)
        raw_gain[i] = float((np.diag(cov) - update[2]).mean())
        updates[i] = update
    profile = raw_gain / variance_scale
    risk_gain = np.full(n, np.nan)
    total = np.full(n, -np.inf)
    shortlist = sorted(eligible, key=lambda i: (-profile[i], i))[:shortlist_size]
    if policy == "gp_alc":
        risk_gain[list(eligible)] = 0.0
        total[list(eligible)] = profile[list(eligible)]
    else:
        before_risk = _risk(means, np.diag(cov), noise, tau,
                            probability_transform, fn_cost, fp_cost)
        # Common random numbers compare candidates with the same Monte Carlo
        # draws. Correlation across all 24 depth cells comes from Cholesky.
        z = rng.standard_normal((fantasies, 24))
        risk_scale = fn_cost * fp_cost / (fn_cost + fp_cost)  # maximum Bayes risk = .8
        for i in shortlist:
            _, solve_cross, after_diag, cholesky, _ = updates[i]
            shifts = (z @ cholesky.T) @ solve_cross
            # Every first-stage fantasy is carried into the second stage.
            # Future innovations are conditionally independent of the first.
            posterior_means = (means[:, None, :] + shifts[None, :, :]).reshape(-1, cov.shape[0])
            after_risk = _risk(posterior_means, after_diag, noise, tau,
                               probability_transform, fn_cost, fp_cost)
            risk_gain[i] = (before_risk - after_risk) / risk_scale
            total[i] = 0.5 * profile[i] + 0.5 * risk_gain[i]
    return profile, risk_gain, total, raw_gain, updates, shortlist


def select_gp_batch(proxy, observed_coordinates, candidate_grid, config, *, policy,
                    min_spacing_m, probability_transform=None):
    """Select two whole CPTs using fixed-kernel GP information gain.

    ``score``/``profile_gain``/``risk_gain`` are initial scores. Their
    ``second_stage_*`` counterparts are conditional marginal scores. SUR only
    computes risk for the ALC-shortlisted candidates; unscored entries are NaN
    in exported tables, never misleading zeros.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown v2 acquisition policy: {policy}")
    if not np.isfinite(min_spacing_m) or min_spacing_m < 0:
        raise ValueError("min_spacing_m must be finite and nonnegative")
    grid = _validate_grid(candidate_grid, config)
    observed = _coordinates(observed_coordinates)
    candidates = _coordinates(grid)
    if len(candidates) < 2 or set(observed.location_id) & set(candidates.location_id):
        raise ValueError("Two unobserved candidate locations are required")
    if proxy.observed is None or set(proxy.observed.location_id.astype(str)) != set(observed.location_id):
        raise ValueError("Proxy must be conditioned on exactly the revealed locations")
    if policy == "gp_threshold":
        chosen, scores = select_batch(observed, candidates, proxy.predict(grid), config,
                                      policy="threshold_uncertainty", min_spacing_m=min_spacing_m)
        scores["profile_gain"] = np.nan
        scores["risk_gain"] = np.nan
        scores["selection_model_id"] = "gp"
        scores["acquisition_method"] = policy
        return chosen, scores

    mean, cov, noise = _latent_posterior(proxy, grid)
    ids = candidates.location_id.tolist()
    groups = [np.flatnonzero(grid.location_id.eq(location)) for location in ids]
    xy = candidates[["x_m", "y_m"]].to_numpy(float)
    distance = _nearest(xy, observed[["x_m", "y_m"]].to_numpy(float))
    settings = _settings(config)
    rng = np.random.default_rng(int(settings.get("seed", config.get("split", {}).get("seed", 20261013))))
    variance_scale = max(float(np.diag(cov).mean()), 1e-12)
    stage1 = _stage_scores(cov, mean[None, :], groups, list(range(len(ids))), noise,
                           variance_scale, config, probability_transform, policy=policy, rng=rng)
    profile1, risk1, total1, raw1, updates1, shortlist1 = stage1
    first = min(range(len(ids)), key=lambda i: (-total1[i], ids[i]))
    cross, solve_cross, after_diag, cholesky, jitter1 = updates1[first]
    after_cov = cov - cross @ solve_cross
    after_cov = (after_cov + after_cov.T) * 0.5
    after_cov.flat[::len(after_cov) + 1] = after_diag
    if policy == "gp_sur":
        z = rng.standard_normal((int(settings.get("fantasies", 16)), 24))
        after_means = mean[None, :] + (z @ cholesky.T) @ solve_cross
    else:
        after_means = mean[None, :]
    distances_to_first = np.linalg.norm(xy - xy[first], axis=1)
    available = [i for i in range(len(ids)) if i != first]
    eligible = [i for i in available if distances_to_first[i] >= min_spacing_m]
    fallback = not eligible
    if fallback:
        # Exactly the v1 fallback: maximize distance first, then marginal
        # score, then stable ID. Score all distance ties before selecting.
        furthest = max(distances_to_first[i] for i in available)
        eligible = [i for i in available if distances_to_first[i] == furthest]
    stage2 = _stage_scores(after_cov, after_means, groups, available, noise,
                           variance_scale, config, probability_transform, policy="gp_alc", rng=rng)
    profile2, risk2, total2, raw2, updates2, _ = stage2
    shortlist2 = available
    if policy == "gp_sur":
        # Shortlist *after* spacing, so none of the eight expensive fantasy
        # evaluations are spent on candidates that cannot be selected.
        stage2 = _stage_scores(after_cov, after_means, groups, eligible, noise,
                               variance_scale, config, probability_transform, policy=policy, rng=rng)
        p2, risk2, total2, _, updates_sur, shortlist2 = stage2
        updates2.update(updates_sur)
    second = min(eligible, key=lambda i: (-total2[i], ids[i]))

    scores = candidates.copy()
    scores["distance_to_observed_m"] = distance
    scores["distance_to_first_m"] = distances_to_first
    scores["profile_gain"] = profile1
    scores["risk_gain"] = risk1
    scores["integrated_variance_gain"] = raw1
    scores["base_score"] = profile1
    scores["score"] = np.where(np.isfinite(total1), total1, np.nan)
    scores["total_score"] = scores.score
    scores["second_stage_profile_gain"] = profile2
    scores["second_stage_risk_gain"] = risk2
    scores["second_stage_integrated_variance_gain"] = raw2
    scores["second_stage_score"] = np.where(np.isfinite(total2), total2, np.nan)
    scores["initial_shortlisted"] = [i in shortlist1 for i in range(len(ids))]
    scores["second_stage_shortlisted"] = [i in shortlist2 for i in range(len(ids))]
    scores["eligible_for_second"] = (distances_to_first >= min_spacing_m)
    scores.loc[first, "eligible_for_second"] = False
    scores["selected"] = False
    scores["selection_order"] = pd.Series(pd.NA, index=scores.index, dtype="Int64")
    scores["selection_score"] = np.nan
    scores["selection_profile_gain"] = np.nan
    scores["selection_risk_gain"] = np.nan
    scores["spacing_fallback"] = False
    for order, i, profile, risk, total in [(1, first, profile1, risk1, total1), (2, second, profile2, risk2, total2)]:
        scores.loc[i, ["selected", "selection_order", "selection_score", "selection_profile_gain", "selection_risk_gain"]] = [True, order, total[i], profile[i], risk[i]]
    scores.loc[second, "spacing_fallback"] = fallback
    scores["selected_rank"] = scores.selection_order
    scores["spacing_relaxed"] = scores.spacing_fallback
    scores["selection_model_id"] = "gp"
    scores["acquisition_method"] = policy
    scores["observation_noise_variance"] = noise
    scores["initial_mean_latent_variance"] = variance_scale
    scores["solve_jitter"] = max(jitter1, updates2[second][-1])
    scores["reference_cells"] = len(grid)
    scores["profile_cells"] = 24
    scores["fantasies_per_candidate"] = int(settings.get("fantasies", 16)) if policy == "gp_sur" else 0
    scores["risk_normalizer"] = 0.8 if policy == "gp_sur" else np.nan
    return [ids[first], ids[second]], scores
