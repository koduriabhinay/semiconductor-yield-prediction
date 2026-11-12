"""End-to-end training: CV evaluation, final model, root-cause analysis,
excursion monitor. Saves artifacts/ and metrics.json with the REAL numbers
produced by this run — nothing is hardcoded.

Usage:  python scripts/train.py [--artifacts artifacts] [--data data]
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secom_yield.data import RANDOM_STATE, download_secom, load_secom
from secom_yield.excursion import (
    ExcursionMonitor,
    order_by_time,
    split_reference_monitor,
)
from secom_yield.modeling import evaluate_cv, fit_final_gbm
from secom_yield.rootcause import (
    mann_whitney_screen,
    root_cause_candidates,
    shap_attribution,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data)
    if not (data_dir / "secom.data").exists():
        print("Downloading SECOM dataset from UCI ...")
        download_secom(data_dir)
    X, y, run_time = load_secom(data_dir)
    print(f"Loaded SECOM: X={X.shape}, fails={int(y.sum())}, passes={int((1 - y).sum())}")

    # 1) 5-fold stratified CV: GBM vs logistic-regression baseline.
    print("Running 5-fold stratified CV (this takes a few minutes) ...")
    metrics = evaluate_cv(X, y)
    gbm_m = metrics["models"]["gbm"]
    lr_m = metrics["models"]["logreg"]
    print(f"GBM    ROC-AUC: {gbm_m['mean']:.4f} +- {gbm_m['std']:.4f}  {gbm_m['fold_scores']}")
    print(f"LogReg ROC-AUC: {lr_m['mean']:.4f} +- {lr_m['std']:.4f}  {lr_m['fold_scores']}")
    print(f"GBM implementation: {metrics['gbm_implementation']}")

    # 2) Final model on all data.
    pipe = fit_final_gbm(X, y)
    prep = pipe.named_steps["prep"]
    feature_names = list(prep.get_feature_names_out())
    print(f"Signals kept after leak-free selection: {len(feature_names)} "
          f"(dropped {len(prep.dropped_columns_)})")

    joblib.dump(pipe, artifacts / "gbm.joblib")
    joblib.dump(prep, artifacts / "prep.joblib")
    (artifacts / "feature_names.json").write_text(json.dumps(list(X.columns)))

    # 3) Attribution + root-cause screen on imputed full data.
    print("Computing feature attribution ...")
    attribution = shap_attribution(pipe, X, y, max_samples=500)
    print(f"Attribution method: {attribution.attrs.get('method')}")
    metrics["attribution_method"] = attribution.attrs.get("method")

    X_imp = pd.DataFrame(prep.transform(X), columns=feature_names)
    print("Running Mann-Whitney U screen with BH-FDR ...")
    candidates = root_cause_candidates(X_imp, y, attribution, top_n=25)
    n_sig = int((mann_whitney_screen(X_imp, y)["significant"]).sum())
    print(f"BH-significant signals (FDR 5%): {n_sig}; reporting top 25 by attribution.")
    candidates.to_csv(artifacts / "root_cause_candidates.csv", index=False)
    metrics["n_bh_significant"] = n_sig
    metrics["n_signals_kept"] = len(feature_names)

    # 4) Excursion monitor: reference = first half of time-ordered runs.
    Xo, yo = order_by_time(X_imp, y, run_time)
    X_ref, y_ref, X_mon, y_mon = split_reference_monitor(Xo, yo, ref_frac=0.5)
    monitor = ExcursionMonitor().fit(X_ref, attribution)
    joblib.dump(monitor, artifacts / "monitor.joblib")
    scored = monitor.score(X_mon)
    n_anom = int(scored["is_anomaly"].sum())
    n_breach = int((scored["n_chart_breaches"] > 0).sum())
    print(f"Monitored runs: {len(X_mon)} | IsolationForest anomalies: {n_anom} | "
          f"runs with 3-sigma breaches: {n_breach}")
    metrics["excursion"] = {
        "n_monitored": len(X_mon),
        "n_reference": len(X_ref),
        "n_isolation_anomalies": n_anom,
        "n_runs_with_chart_breaches": n_breach,
        "top_signals": monitor.top_signals_,
        "control_limits": monitor.control_limits_,
    }
    scored.to_csv(artifacts / "excursion_scores.csv", index=False)

    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Artifacts + metrics.json written to {artifacts}/")


if __name__ == "__main__":
    main()
