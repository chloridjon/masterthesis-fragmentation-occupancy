"""Summary diagnostics: retain complete seeds as independent units."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

PAIR_KEYS = ["reference_representation", "candidate_representation", "metric", "H_target"]

def correlation(a, b):
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 4 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

def adaptive_pair_intervals(summary, pairs, settings, seed):
    for field in ("initial_draws", "batch_draws", "maximum_draws", "consecutive_stable_batches"):
        if not isinstance(settings[field], int) or settings[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    for field in ("correlation_tolerance", "difference_scale_tolerance", "loo_correlation_tolerance", "loo_difference_scale_tolerance"):
        if not np.isfinite(settings[field]) or settings[field] < 0:
            raise ValueError(f"{field} must be finite and nonnegative")
    intervals, batches, influence = [], [], []
    for key, part in pairs.groupby(PAIR_KEYS, sort=True):
        identity = dict(zip(PAIR_KEYS, key))
        a, b, d = (part[c].to_numpy(float) for c in ("reference", "candidate", "difference"))
        blocks = part["seed"].to_numpy()
        seeds = np.sort(np.unique(blocks))
        indices = {s: np.flatnonzero(blocks == s) for s in seeds}
        local_seed = int(hashlib.sha256(repr(tuple(key)).encode()).hexdigest()[:8], 16) ^ seed
        rng = np.random.default_rng(local_seed)
        # Normalize differences by a fixed, nonnegative reference scale, never by a signed gap.
        scale = max(float(np.median(np.abs(a))), float(np.quantile(np.abs(a), .95)), 1e-12)
        original = np.array([correlation(a,b), np.median(d)])
        loo = []
        for omitted in seeds:
            keep = blocks != omitted
            estimates = np.array([correlation(a[keep], b[keep]), np.median(d[keep])])
            loo.append(estimates)
            influence.append({**identity, "omitted_seed": int(omitted), "n_seed_blocks": len(seeds)-1,
                              "spearman": estimates[0], "median_difference": estimates[1]})
        loo = np.array(loo)
        corr_change = float(np.nanmax(np.abs(loo[:,0]-original[0]))) if np.isfinite(original[0]) else np.nan
        gap_change = float(np.max(np.abs(loo[:,1]-original[1])) / scale)
        correlations, medians = [], []
        previous = None
        stable_count = 0
        converged = False
        while len(medians) < settings["maximum_draws"]:
            target = min(settings["initial_draws"] if not medians else len(medians)+settings["batch_draws"], settings["maximum_draws"])
            for _ in range(target-len(medians)):
                chosen = rng.choice(seeds, len(seeds), replace=True)
                ix = np.concatenate([indices[s] for s in chosen])
                correlations.append(correlation(a[ix],b[ix]))
                medians.append(float(np.median(d[ix])))
            finite = np.array(correlations)[np.isfinite(correlations)]
            cq = np.quantile(finite, [.025,.5,.975]) if len(finite) else np.full(3,np.nan)
            dq = np.quantile(medians,[.025,.5,.975])
            current = np.r_[cq,dq]
            corr_delta = float(np.max(np.abs(current[:3]-previous[:3]))) if previous is not None and len(finite) else np.nan
            gap_delta = float(np.max(np.abs(current[3:]-previous[3:]))/scale) if previous is not None else np.nan
            # Undefined rank correlation (constant metric) is not a convergence failure.
            rank_ok = not len(finite) or (np.isfinite(corr_delta) and corr_delta <= settings["correlation_tolerance"])
            ok = previous is not None and rank_ok and gap_delta <= settings["difference_scale_tolerance"]
            stable_count = stable_count+1 if ok else 0
            converged = stable_count >= settings["consecutive_stable_batches"]
            batches.append({**identity,"bootstrap_seed":local_seed,"draws":len(medians),"finite_correlation_draws":len(finite),
                           "correlation_max_change":corr_delta,"scaled_difference_max_change":gap_delta,
                           "consecutive_stable_batches":stable_count,"converged":converged})
            if converged: break
            previous = current
        intervals.append({**identity,"spearman_ci_low":cq[0] if converged else np.nan,"spearman_ci_high":cq[2] if converged else np.nan,
                          "median_difference_ci_low":dq[0] if converged else np.nan,"median_difference_ci_high":dq[2] if converged else np.nan,
                          "bootstrap_draws":len(medians),"bootstrap_converged":converged,
                          "raw_spearman_ci_low":cq[0],"raw_spearman_ci_high":cq[2],
                          "raw_median_difference_ci_low":dq[0],"raw_median_difference_ci_high":dq[2],
                          "n_seed_blocks":len(seeds),"loo_correlation_tolerance":settings["loo_correlation_tolerance"],"loo_max_spearman_change":corr_change,"loo_max_scaled_difference_change":gap_change,
                          "seed_influence_status":"sensitive" if (np.isfinite(corr_change) and corr_change>settings["loo_correlation_tolerance"]) or gap_change>settings["loo_difference_scale_tolerance"] else "stable"})
        print(f"Bootstrap {key}: {len(medians)} draws, converged={converged}", flush=True)
    return summary.merge(pd.DataFrame(intervals),on=PAIR_KEYS,validate="one_to_one"), pd.DataFrame(batches), pd.DataFrame(influence)

def factor_seed_influence(data, config, metric_summary, factor_summary, settings):
    if not np.isfinite(settings["maximum_main_share_change"]) or settings["maximum_main_share_change"] < 0:
        raise ValueError("maximum_main_share_change must be finite and nonnegative")
    keys=["rho_index","rho","H_target","metric"]
    estimates=factor_summary(metric_summary(data),config)
    records=[]
    for omitted in sorted(data["seed"].unique()):
        result=factor_summary(metric_summary(data[data["seed"].ne(omitted)]),config)
        result["omitted_seed"]=int(omitted)
        records.append(result)
    loo=pd.concat(records,ignore_index=True)
    diagnostics=[]
    columns=["share_eta","share_k","share_lambda"]
    for _, row in estimates.iterrows():
        part=loo
        for key in keys: part=part[part[key].eq(row[key])]
        base=row.reindex(columns).to_numpy(float)
        values=part.reindex(columns=columns).to_numpy(float)
        finite=np.isfinite(values).all(axis=1)
        change=float(np.max(np.abs(values[finite]-base))) if finite.any() and np.isfinite(base).all() else np.nan
        dominant_consistency=float(np.mean(np.argmax(values[finite],axis=1)==np.argmax(base))) if finite.any() and np.isfinite(base).all() else np.nan
        direction = {}
        for factor in ("eta", "k", "lambda"):
            column = f"delta_{factor}_high_minus_low"
            base_delta = row.get(column, np.nan)
            delta = part.reindex(columns=[column])[column].to_numpy(float)
            valid = np.isfinite(delta)
            direction[f"loo_{factor}_direction_agreement"] = float(np.mean(np.sign(delta[valid]) == np.sign(base_delta))) if valid.any() and np.isfinite(base_delta) else np.nan
        diagnostics.append({**{k:row[k] for k in keys}, **direction,"n_loo_defined":int(finite.sum()),
                            "loo_max_main_share_change":change,"loo_dominant_factor_agreement":dominant_consistency,
                            "seed_influence_status": "unavailable" if not row["sufficient_replication"] or not np.isfinite(base).all() else
                            ("stable" if finite.all() and change<=settings["maximum_main_share_change"] and dominant_consistency==1 else "sensitive")})
    return estimates.merge(pd.DataFrame(diagnostics),on=keys,validate="one_to_one"),loo


def summary_provenance(settings):
    return {
        "summary_settings": settings,
        "stability_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
