#!/usr/bin/env python3
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance

FEATURES_FILE = "feature_list.json"

def load_features():
    if not os.path.exists(FEATURES_FILE):
        print(f"ERROR: {FEATURES_FILE} not found. Train once to write it.", file=sys.stderr)
        sys.exit(1)
    with open(FEATURES_FILE, "r") as f:
        return json.load(f)

def load_xy(csv_path, target_col, filter_ws_for_outcomes=False):
    df = pd.read_csv(csv_path)
    feats = load_features()
    if filter_ws_for_outcomes and "y_shape" in df.columns:
        df = df[df["y_shape"].fillna(0).astype(int) == 1].copy()
    df = df[df[target_col].isin([0,1])].copy()
    if df.empty:
        print(f"No rows with {target_col} in {{0,1}} after filtering.", file=sys.stderr)
        sys.exit(1)
    X = df[feats].astype(float).fillna(0.0).to_numpy()
    y = df[target_col].astype(int).to_numpy()
    return df, X, y, feats

def ensure_outdir(p):
    os.makedirs(p, exist_ok=True)
    return p

def plot_bar(names, vals, out_png, title):
    order = np.argsort(vals)[::-1]
    names = np.array(names)[order]
    vals = np.array(vals)[order]
    plt.figure(figsize=(8, max(3, 0.25*len(names))))
    plt.barh(range(len(names)), vals)
    plt.yticks(range(len(names)), names)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def cmd_importance(args):
    model = load(args.model)
    filter_ws = args.filter_w and (args.target_column != "y_shape")
    df, X, y, feats = load_xy(args.csv, args.target_column, filter_ws_for_outcomes=filter_ws)

    outdir = ensure_outdir(args.out)
    # In-sample AUC (quick sanity check)
    try:
        proba = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, proba)
        print(f"In-sample AUC({args.target_column}): {auc:.3f}  (N={len(y)})")
    except Exception:
        proba = None
        print("Model has no predict_proba; skipping in-sample AUC.")

    # Tree/impurity feature importances
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        tab = pd.DataFrame({"feature": feats, "importance": fi})
        tab = tab.sort_values("importance", ascending=False)
        csv_p = os.path.join(outdir, f"{args.name}_feat_importance_impurity.csv")
        tab.to_csv(csv_p, index=False)
        print(f"Saved impurity importances → {csv_p}")
        plot_bar(tab["feature"].tolist(), tab["importance"].to_numpy(),
                 os.path.join(outdir, f"{args.name}_feat_impurity.png"),
                 f"{args.name}: Feature importance (impurity)")
    else:
        print("Model lacks feature_importances_; skipping impurity importances.")

    # Permutation importance
    nrep = args.permutation_repeats
    if nrep > 0:
        print(f"Running permutation importance (n_repeats={nrep})...")
        r = permutation_importance(model, X, y, n_repeats=nrep, random_state=42, scoring="roc_auc")
        tabp = pd.DataFrame({
            "feature": feats,
            "mean_importance": r.importances_mean,
            "std_importance": r.importances_std
        }).sort_values("mean_importance", ascending=False)
        csv_pp = os.path.join(outdir, f"{args.name}_feat_importance_permutation.csv")
        tabp.to_csv(csv_pp, index=False)
        print(f"Saved permutation importances → {csv_pp}")
        plot_bar(tabp["feature"].tolist(), tabp["mean_importance"].to_numpy(),
                 os.path.join(outdir, f"{args.name}_feat_perm.png"),
                 f"{args.name}: Feature importance (permutation)")

def cmd_roc(args):
    model = load(args.model)
    filter_ws = args.filter_w and (args.target_column != "y_shape")
    df, X, y, feats = load_xy(args.csv, args.target_column, filter_ws_for_outcomes=filter_ws)
    proba = model.predict_proba(X)[:,1]
    fpr, tpr, thr = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    outdir = ensure_outdir(args.out)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.title(f"ROC {args.name} (AUC={auc:.3f})")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.tight_layout()
    png = os.path.join(outdir, f"{args.name}_roc.png")
    plt.savefig(png, dpi=160); plt.close()
    print(f"Saved ROC plot → {png}")

def main():
    ap = argparse.ArgumentParser(description="Model inspection (feature importance, ROC)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("importance", help="Feature importance (impurity + permutation)")
    p1.add_argument("--model", required=True, help="Path to model .pkl (e.g., model_shape.pkl)")
    p1.add_argument("--csv", required=True, help="w_with_features.csv")
    p1.add_argument("--target-column", required=True, choices=["y_shape","y_breakout","y_reachtarget"])
    p1.add_argument("--name", required=True, help="Short name used in output files")
    p1.add_argument("--out", default="reports", help="Output folder")
    p1.add_argument("--permutation-repeats", type=int, default=10)
    p1.add_argument("--filter-w", action="store_true",
                    help="For outcome models, filter to y_shape==1 (matches training)")
    p1.set_defaults(func=cmd_importance)

    p2 = sub.add_parser("roc", help="ROC curve on training data (quick sanity check)")
    p2.add_argument("--model", required=True)
    p2.add_argument("--csv", required=True)
    p2.add_argument("--target-column", required=True, choices=["y_shape","y_breakout","y_reachtarget"])
    p2.add_argument("--name", required=True)
    p2.add_argument("--out", default="reports")
    p2.add_argument("--filter-w", action="store_true")
    p2.set_defaults(func=cmd_roc)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
