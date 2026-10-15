"""LightGBM pipeline for the ATLAS Higgs ML Challenge 2014 data (v2 rewrite).

Expected input: the Kaggle-style training table, i.e. the original CERN Open
Data columns (EventId, DER_*, PRI_*, Weight, Label in {'s','b'}).

If you are using the full CERN Open Data file (818,238 events) that carries
KaggleSet / KaggleWeight columns, run it through load_kaggle_training() first.
That keeps the 250,000 'KaggleSet == t' events and swaps in KaggleWeight, so the
class weight totals (about 692 signal, about 411,000 background) match the
challenge normalization and your AMS values are comparable to the leaderboard.

Usage:
    train_df = load_kaggle_training("atlas-higgs-challenge-2014-v2.csv")
    model, thr, val = train_lgbm(train_df)
    plot_ams_curve(val, x="rejected")
    plot_ams_curve(val, x="threshold")
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_kaggle_training(path):
    """Load the CERN Open Data CSV and return the challenge training set."""
    df = pd.read_csv(path)
    if "KaggleSet" in df.columns:
        df = df[df["KaggleSet"] == "t"].copy()
        df["Weight"] = df["KaggleWeight"]
    return df


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


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
def balance_weights(y, w):
    """Training weights: signal and background each sum to the same total,
    then scaled to mean 1. The raw challenge weights make signal about 0.2% of
    the total weight, which (a) pushes predicted probabilities toward zero so
    0.5-0.95 cuts are meaningless and (b) makes hessians so small that
    LightGBM's min_sum_hessian_in_leaf can block splits."""
    y = np.asarray(y)
    w = np.asarray(w, dtype=float).copy()
    w[y == 1] *= w[y == 0].sum() / w[y == 1].sum()
    return w / w.mean()


def rescale_to_totals(y_sub, w_sub, y_full, w_full):
    """Eq. 31 of Adam-Bourdarios et al. 2015: scale each class's weights in a
    subset so the class totals match the full sample."""
    y_sub, y_full = np.asarray(y_sub), np.asarray(y_full)
    w_out = np.asarray(w_sub, dtype=float).copy()
    w_full = np.asarray(w_full, dtype=float)
    for c in (0, 1):
        w_out[y_sub == c] *= w_full[y_full == c].sum() / w_out[y_sub == c].sum()
    return w_out


# ---------------------------------------------------------------------------
# AMS
# ---------------------------------------------------------------------------
def ams(s, b, b_reg=10.0):
    """Approximate median significance (works on scalars or arrays)."""
    s = np.asarray(s, dtype=float)
    b = np.asarray(b, dtype=float)
    radicand = 2.0 * ((s + b + b_reg) * np.log1p(s / (b + b_reg)) - s)
    out = np.sqrt(np.clip(radicand, 0.0, None))
    out = np.where((s < 0) | (b < 0), np.nan, out)
    return float(out) if out.ndim == 0 else out


def ams_scan(y, p, w_ams, b_reg=10.0):
    """AMS for every possible cut 'select events with p >= t', computed in one
    pass with cumulative sums. Ties in p are handled (cuts only fall between
    distinct prediction values)."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    w = np.asarray(w_ams, dtype=float)

    order = np.argsort(-p, kind="stable")
    p, y, w = p[order], y[order], w[order]

    s_cum = np.cumsum(w * (y == 1))
    b_cum = np.cumsum(w * (y == 0))
    n_sel = np.arange(1, len(p) + 1)

    # keep only the last row of each run of identical predictions
    last = np.r_[p[1:] != p[:-1], True]
    thr, s, b, n_sel = p[last], s_cum[last], b_cum[last], n_sel[last]

    return {
        "threshold": thr,
        "ams": ams(s, b, b_reg),
        "rejected": 100.0 * (1.0 - n_sel / len(p)),  # percent of events cut
    }


def best_ams_threshold(y_true, p, w, b_reg=10.0):
    """Return (best_threshold, best_ams). Select with p >= best_threshold.
    w must already be rescaled to the full-sample totals (see train_lgbm)."""
    scan = ams_scan(y_true, p, w, b_reg)
    i = int(np.nanargmax(scan["ams"]))
    return scan["threshold"][i], scan["ams"][i]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_lgbm(df, seed=42, feature_group=None):
    """Train on all features, or only one feature family ("DER" or "PRI").

    Returns (model, best_threshold, val) where val holds the held-out split:
      y, p       labels and predicted signal probability
      w          original event weights (physical ROC)
      w_ams      weights rescaled to full-sample class totals (AMS)
    """
    X, y, w = prep(df, feature_group=feature_group)
    Xtr, Xva, ytr, yva, wtr, wva = train_test_split(
        X, y, w, test_size=0.2, stratify=y, random_state=seed
    )

    # Balanced weights for fitting and early stopping
    wtr_bal = balance_weights(ytr.values, wtr.values)
    wva_bal = balance_weights(yva.values, wva.values)

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
    dtr = lgb.Dataset(Xtr, ytr, weight=wtr_bal)
    dva = lgb.Dataset(Xva, yva, weight=wva_bal, reference=dtr)

    model = lgb.train(
        params,
        dtr,
        num_boost_round=2000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    p = model.predict(Xva, num_iteration=model.best_iteration)

    # The validation split holds 20% of events, so rescale each class's weights
    # up to the full-sample totals before computing AMS.
    w_ams = rescale_to_totals(yva.values, wva.values, y.values, w.values)
    thr, score = best_ams_threshold(yva.values, p, w_ams)
    frac_sel = 100.0 * (p >= thr).mean()
    print(
        f"best iteration: {model.best_iteration}, threshold: {thr:.3f}, "
        f"AMS: {score:.3f}, events selected: {frac_sel:.1f}%"
    )

    val = {"y": yva.values, "p": p, "w": wva.values, "w_ams": w_ams}
    return model, thr, val


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_roc(val, weighted=True, ax=None):
    """ROC curve on the held-out validation split.

    weighted=True uses the original event weights (the physically meaningful
    curve); weighted=False treats every event equally."""
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


def plot_roc_paper_axes(val, weighted=False, ax=None, label="Ours (LightGBM)"):
    """ROC with x = signal efficiency (TPR), y = background rejection (1 - FPR).
    Same curve as plot_roc, different axes."""
    w = val["w"] if weighted else None
    fpr, tpr, _ = roc_curve(val["y"], val["p"], sample_weight=w)
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    ax.plot(tpr, 1.0 - fpr, label=label)
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    return ax


def plot_ams_curve(val, x="rejected", ax=None, label="AMS", b_reg=10.0):
    """AMS versus selection cut on the validation split.

    x="rejected":  % of events rejected (cut tightens left to right)
    x="threshold": probability cutoff (select p >= threshold)
    """
    if x not in ("rejected", "threshold"):
        raise ValueError("x must be 'rejected' or 'threshold'")

    scan = ams_scan(val["y"], val["p"], val["w_ams"], b_reg)
    xs = scan[x]
    scores = scan["ams"]

    order = np.argsort(xs)
    xs, scores = xs[order], scores[order]
    i = int(np.nanargmax(scores))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    (line,) = ax.plot(xs, scores, label=label)
    ax.plot(xs[i], scores[i], "o", color=line.get_color(),
            label=f"{label} max = {scores[i]:.3f}")

    ax.set_xlabel("% of events rejected" if x == "rejected" else "Probability threshold")
    ax.set_ylabel("AMS")
    ax.set_title("AMS vs. selection cut, validation split")
    ax.legend(loc="lower center")
    return ax


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
def make_submission(model, df_test, threshold, path="submission.csv"):
    """Kaggle-style file: EventId, RankOrder (1 = most signal-like), Class."""
    X, _, _ = prep(df_test)
    X = X[model.feature_name()]  # same columns, same order as training
    p = model.predict(X, num_iteration=model.best_iteration)
    rank = (-p).argsort(kind="stable").argsort() + 1
    out = pd.DataFrame(
        {
            "EventId": df_test["EventId"].values,
            "RankOrder": rank,
            "Class": np.where(p >= threshold, "s", "b"),
        }
    ).sort_values("EventId")
    out.to_csv(path, index=False)
    return out


# ---------------------------------------------------------------------------
# Example run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_df = load_kaggle_training("atlas-higgs-challenge-2014-v2.csv")
    model, thr, val = train_lgbm(train_df)

    plot_roc(val)
    plot_ams_curve(val, x="rejected")
    plot_ams_curve(val, x="threshold")

    # Compare models trained on derived (DER) and primitive (PRI) feature families.
    ams_by_group = {}
    for feature_group in ("DER", "PRI"):
        _, _, ams_by_group[feature_group] = train_lgbm(
            train_df, feature_group=feature_group
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    for feature_group, group_val in ams_by_group.items():
        plot_ams_curve(group_val, x="rejected", ax=ax, label=f"{feature_group}-only")
    ax.set_title("AMS by feature family, validation split")
    plt.show()
