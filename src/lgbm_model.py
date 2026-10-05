"""LightGBM pipeline for the ATLAS Higgs ML Challenge 2014 data.

Assumes a DataFrame with the original CERN Open Data columns:
EventId, DER_*, PRI_*, Weight, Label ('s' or 'b').
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


def ams(s, b, b_reg=10.0):
    """Approximate Median Significance (challenge metric)."""
    return np.sqrt(2.0 * ((s + b + b_reg) * np.log(1.0 + s / (b + b_reg)) - s))


def prep(df, feature_group=None):
    """Return features X, binary labels y, weights w. -999.0 becomes NaN
    (LightGBM routes missing values natively, so no imputation needed).

    feature_group can be "DER" or "PRI" to use only that feature family.
    """
    if feature_group not in (None, "DER", "PRI"):
        raise ValueError("feature_group must be None, 'DER', or 'PRI'")
    prefixes = ("DER_", "PRI_") if feature_group is None else (f"{feature_group}_",)
    feats = [c for c in df.columns if c.startswith(prefixes)]
    if not feats:
        raise ValueError(f"No features found for feature_group={feature_group!r}")
    X = df[feats].replace(-999.0, np.nan)
    y = (df["Label"] == "s").astype(int) if "Label" in df else None
    w = df["Weight"] if "Weight" in df else None
    return X, y, w


def best_ams_threshold(y_true, p, w, grid=None):
    """Scan probability thresholds; return (best_threshold, best_ams).
    w must already be rescaled to the full-sample total (see train_lgbm)."""
    grid = np.linspace(0.5, 0.95, 91) if grid is None else grid
    best = (0.5, -np.inf)
    for t in grid:
        sel = p > t
        s = w[(y_true == 1) & sel].sum()
        b = w[(y_true == 0) & sel].sum()
        score = ams(s, b)
        if score > best[1]:
            best = (t, score)
    return best


def train_lgbm(df, seed=42, feature_group=None):
    """Train on all features, or only one feature family ("DER" or "PRI")."""
    X, y, w = prep(df, feature_group=feature_group)
    Xtr, Xva, ytr, yva, wtr, wva = train_test_split(
        X, y, w, test_size=0.2, stratify=y, random_state=seed
    )

    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        metric="auc",
        seed=seed,
        verbose=-1,
    )
    dtr = lgb.Dataset(Xtr, ytr, weight=wtr)
    dva = lgb.Dataset(Xva, yva, weight=wva, reference=dtr)

    model = lgb.train(
        params,
        dtr,
        num_boost_round=2000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    # Validation weights cover 20% of events, so rescale to the full sample
    p = model.predict(Xva, num_iteration=model.best_iteration)
    # Renormalize validation weights per class (Eq. 31 in Adam-Bourdarios et al. 2015):
    # signal and background validation weights are each scaled up to the full-sample totals
    w_ams = wva.values.astype(float).copy()
    for c in (0, 1):
        w_ams[yva.values == c] *= w[y == c].sum() / wva[yva == c].sum()
    thr, score = best_ams_threshold(yva.values, p, w_ams)
    print(f"best iteration: {model.best_iteration}, threshold: {thr:.3f}, AMS: {score:.3f}")
    val = {"y": yva.values, "p": p, "w": wva.values, "w_ams": w_ams}
    return model, thr, val


def plot_roc(val, weighted=True, ax=None):
    """ROC curve on the held-out validation split.

    weighted=True uses the event Weight column (matches the AUC LightGBM
    reports during training); weighted=False treats every event equally.
    """
    w = val["w"] if weighted else None
    fpr, tpr, _ = roc_curve(val["y"], val["p"], sample_weight=w)
    auc = roc_auc_score(val["y"], val["p"], sample_weight=w)

    ax = ax or plt.subplots(figsize=(5, 5))[1]
    label = f"LightGBM ({'weighted' if weighted else 'unweighted'}), AUC = {auc:.4f}"
    ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_xlabel("False positive rate (background efficiency)")
    ax.set_ylabel("True positive rate (signal efficiency)")
    ax.set_title("ROC curve, validation split")
    ax.legend(loc="lower right")
    return ax


def make_submission(model, df_test, threshold, path="submission.csv"):
    """Kaggle-style file: EventId, RankOrder (1 = most signal-like), Class."""
    X, _, _ = prep(df_test)
    p = model.predict(X, num_iteration=model.best_iteration)
    # argsort of argsort gives ranks; negate so the highest score gets rank 1
    rank = (-p).argsort().argsort() + 1
    out = pd.DataFrame(
        {
            "EventId": df_test["EventId"].values,
            "RankOrder": rank,
            "Class": np.where(p > threshold, "s", "b"),
        }
    ).sort_values("EventId")
    out.to_csv(path, index=False)
    return out


def plot_roc_paper_axes(val, weighted=False, ax=None, label="Ours (LightGBM)"):
    """ROC drawn like Baldi et al. 2014 Fig. 7: x = signal efficiency (TPR),
    y = background rejection (1 - FPR). Same curve as plot_roc, different axes."""
    w = val["w"] if weighted else None
    fpr, tpr, _ = roc_curve(val["y"], val["p"], sample_weight=w)
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    ax.plot(tpr, 1.0 - fpr, label=label)
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    return ax


def plot_ams_curve(val, x="rejected", ax=None, label="AMS"):

    y = np.asarray(val["y"])
    p = np.asarray(val["p"])
    w = np.asarray(val["w_ams"])

    # Sort by prediction
    order = np.argsort(p)[::-1]

    y = y[order]
    p = p[order]
    w = w[order]

    # Find unique prediction thresholds
    unique_p, first_idx = np.unique(p, return_index=True)

    # We sorted descending, so reverse
    unique_p = unique_p[::-1]

    scores = []
    rejected = []

    for t in unique_p:

        sel = p >= t

        s = w[(y == 1) & sel].sum()
        b = w[(y == 0) & sel].sum()

        scores.append(ams(s, b))
        rejected.append(100 * (~sel).mean())

    scores = np.asarray(scores)
    rejected = np.asarray(rejected)

    thresholds = unique_p

    i = np.argmax(scores)

    xs = rejected if x == "rejected" else thresholds

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(xs, scores, label=label)

    ax.plot(
        xs[i],
        scores[i],
        "o",
        label=f"{label} max = {scores[i]:.3f}"
    )

    ax.set_xlabel(
        "% of events rejected"
        if x == "rejected"
        else "Probability threshold"
    )

    ax.set_ylabel("AMS")
    ax.set_title("AMS vs. selection cut, validation split")
    ax.legend(loc="lower center")

    return ax