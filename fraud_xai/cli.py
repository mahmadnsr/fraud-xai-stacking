"""Command-line interface:  python -m fraud_xai.cli {check-config|train|verify|list}"""
from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_settings
from .data import load_local_csv, load_synthetic
from .logging_utils import setup_logging
from .models import TrainConfig
from .persist import ArtifactError, list_versions, verify_version
from .schema import ValidationError
from .train import train_all


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fraud_xai")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-config")
    sub.add_parser("list")
    v = sub.add_parser("verify")
    v.add_argument("version")
    t = sub.add_parser("train")
    t.add_argument("--source", choices=["sample", "csv"], default="sample", help="'sample' = SYNTHETIC demo data")
    t.add_argument("--csv", help="file name inside data/ (e.g. creditcard.csv)")
    t.add_argument("--target", default="Class")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--split", choices=["stratified", "chronological"], default="stratified")
    t.add_argument("--n-estimators", type=int, default=200)
    t.add_argument("--cv-folds", type=int, default=5)
    t.add_argument("--sample-rows", type=int, default=10_000)
    a = ap.parse_args(argv)
    try:
        settings = load_settings()
        setup_logging(settings.log_level, settings.logs_dir)
        if a.cmd == "check-config":
            print(f"Configuration OK (APP_ENV={settings.app_env}, auth_disabled={settings.auth_disabled}).")
        elif a.cmd == "list":
            print("\n".join(list_versions(settings.models_dir)) or "No model versions.")
        elif a.cmd == "verify":
            verify_version(settings, a.version)
            print(f"{a.version}: signature and file digests verified.")
        else:
            if a.source == "csv":
                if not a.csv:
                    ap.error("--csv is required with --source csv")
                bundle = load_local_csv(a.csv, settings.data_dir, settings.max_upload_bytes, settings.max_rows, a.target)
            else:
                bundle = load_synthetic(a.sample_rows, seed=a.seed)
            print(f"Dataset: {bundle.label} | rows={bundle.report.n_rows} fraud={bundle.report.class_counts[1]}")
            cfg = TrainConfig(seed=a.seed, cv_folds=a.cv_folds, split_strategy=a.split, n_estimators=a.n_estimators)
            res = train_all(bundle, cfg, settings, progress=print)
            print(f"\nVersion {res.version}. Held-out TEST metrics (threshold chosen on validation):")
            for name, m in res.metrics.items():
                print(f"  {name:22s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f}")
            if bundle.is_synthetic:
                print("NOTE: these numbers come from SYNTHETIC data and do not reflect real-world fraud detection performance.")
    except (ConfigError, ValidationError, ArtifactError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
