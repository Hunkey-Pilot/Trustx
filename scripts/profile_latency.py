"""Local latency profile of the /transactions/analyze pipeline.

Part 1 (default, read-only): times each in-process stage separately - feature
engineering, XGBoost + Isolation Forest, SHAP, behavioral / network queries (read-only
SELECTs on the real database), counterfactual (4 model evaluations) - and the sum.

Part 2 (--http URL): sends real POST /analyze requests to a RUNNING backend and reports
client-side latency and throughput. This WRITES rows (sender ids start with LOADTEST_);
add --cleanup to delete exactly those rows (and their audit entries) afterwards.

This is a small single-machine benchmark. It says nothing about production scale.

    backend\\.venv\\Scripts\\python.exe scripts\\profile_latency.py --n 200
    backend\\.venv\\Scripts\\python.exe scripts\\profile_latency.py --n 100 --http http://127.0.0.1:8000 --workers 4 --cleanup
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def summarize(label: str, values: list[float]) -> dict:
    return {
        "stage": label,
        "mean_ms": round(statistics.mean(values) * 1000, 3),
        "p95_ms": round(percentile(values, 0.95) * 1000, 3),
        "max_ms": round(max(values) * 1000, 3),
    }


def payload(index: int) -> dict:
    return {
        "step": 100 + (index % 500), "type": "TRANSFER" if index % 2 else "CASH_OUT",
        "amount": 100.0 + index, "nameOrig": f"LOADTEST_S{index}", "oldbalanceOrg": 5000.0 + index,
        "newbalanceOrig": 4900.0, "nameDest": f"LOADTEST_D{index % 40}", "oldbalanceDest": 300.0,
        "newbalanceDest": 400.0 + index, "historical_transactions": [],
    }


def profile_stages(n: int) -> list[dict]:
    from app.db.session import SessionLocal
    from app.schemas.transactions import TransactionAnalyzeRequest
    from app.services.behavioral_engine import BehavioralEngine
    from app.services.counterfactual_engine import CounterfactualEngine
    from app.services.evidence_engine import EvidenceEngine
    from app.services.feature_engineering import create_model_features
    from app.services.model_loader import ModelLoader
    from app.services.model_service import ModelService
    from app.services.network_engine import NetworkEngine
    from app.services.risk_engine import RiskEngine
    from app.services.shap_service import ShapService

    loaded = ModelLoader().load()
    model_service, shap_service = ModelService(loaded), ShapService(loaded)
    risk, evidence = RiskEngine(), EvidenceEngine()
    behavioral, network, counterfactual = BehavioralEngine(), NetworkEngine(), CounterfactualEngine()
    db = SessionLocal()
    timings: dict[str, list[float]] = {k: [] for k in
                                        ("feature_engineering", "xgboost_and_isolation_forest", "shap", "behavioral_queries",
                                         "network_queries", "counterfactual_4_scenarios", "risk_and_evidence")}
    totals: list[float] = []
    empty = pd.DataFrame(columns=["step", "amount", "nameOrig"])
    now = datetime.now(timezone.utc)
    try:
        for i in range(n + 20):
            body = payload(i)
            request = TransactionAnalyzeRequest.model_validate(body)
            values = request.model_dump(exclude={"historical_transactions"})
            t0 = time.perf_counter()
            features = create_model_features(values, empty, loaded.feature_list)
            t1 = time.perf_counter()
            prediction = model_service.predict(features)
            t2 = time.perf_counter()
            explanation = shap_service.explain(features)
            t3 = time.perf_counter()
            behavior_signals = behavioral.calculate(db, request, "profile", now)
            t4 = time.perf_counter()
            network_signals = network.calculate(db, request, "profile", now)
            t5 = time.perf_counter()
            counterfactual.evaluate("profile", request, prediction.fraud_probability,
                                    loaded.feature_list, model_service)
            t6 = time.perf_counter()
            result = risk.calculate(prediction.fraud_probability, prediction.anomaly_signal)
            evidence.aggregate(prediction.fraud_probability, prediction.anomaly_signal, explanation, result,
                               behavior_signals, network_signals)
            t7 = time.perf_counter()
            if i >= 20:  # warm-up excluded
                for key, value in zip(timings, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6)):
                    timings[key].append(value)
                totals.append(t7 - t0)
    finally:
        db.close()
    rows = [summarize(key, values) for key, values in timings.items()]
    rows.append(summarize("TOTAL (excl. counterfactual, which is a separate request)",
                          [t - c for t, c in zip(totals, timings["counterfactual_4_scenarios"])]))
    rows.append(summarize("TOTAL incl. counterfactual", totals))
    return rows


def post(base: str, key: str, body: dict) -> tuple[int, float]:
    request = urllib.request.Request(
        f"{base}/api/v1/transactions/analyze", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": key}, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response.read()
            return response.status, time.perf_counter() - start
    except Exception:  # noqa: BLE001
        return 0, time.perf_counter() - start


def http_load(base: str, n: int, workers: int) -> dict:
    key = os.getenv("TRUSTX_ANALYST_API_KEY")
    if not key:
        raise SystemExit("TRUSTX_ANALYST_API_KEY is not configured")
    offset = int(time.time()) % 100000 * 1000  # unique sender ids per run (avoids duplicate protection)
    bodies = []
    for i in range(n):
        body = payload(i)
        body["nameOrig"] = f"LOADTEST_{offset + i}"
        bodies.append(body)
    for body in bodies[:5]:
        post(base, key, {**body, "nameOrig": body["nameOrig"] + "W"})  # warm-up
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda b: post(base, key, b), bodies))
    elapsed = time.perf_counter() - start
    latencies = [latency for status, latency in results if status == 200]
    errors = Counter_status([status for status, _ in results])
    return {
        "requests": n, "workers": workers, "ok": len(latencies), "status_counts": errors,
        "wall_seconds": round(elapsed, 3), "throughput_rps": round(len(latencies) / elapsed, 2),
        "mean_ms": round(statistics.mean(latencies) * 1000, 2) if latencies else None,
        "p95_ms": round(percentile(latencies, 0.95) * 1000, 2) if latencies else None,
    }


def Counter_status(values):
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def cleanup() -> int:
    from sqlalchemy import delete

    from app.db.models import AuditLog, TransactionAnalysis
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        ids = [row[0] for row in db.query(TransactionAnalysis.transaction_id).filter(
            TransactionAnalysis.name_orig.like("LOADTEST\\_%", escape="\\")).all()]
        if ids:
            db.execute(delete(AuditLog).where(AuditLog.resource_id.in_(ids)))
            db.execute(delete(TransactionAnalysis).where(TransactionAnalysis.transaction_id.in_(ids)))
            db.commit()
        return len(ids)
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--http", default=None, help="base URL of a running backend (writes LOADTEST_ rows)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cleanup", action="store_true", help="delete LOADTEST_ rows afterwards")
    parser.add_argument("--out", default=str(ROOT / "evaluation" / "results" / "latency_profile.json"))
    args = parser.parse_args()
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    report: dict = {"note": "Single-machine local benchmark; not a production throughput claim.",
                    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "iterations": args.n}
    stages = profile_stages(args.n)
    report["in_process_stages"] = stages
    print(f"{'stage':62s} {'mean ms':>9s} {'p95 ms':>9s}")
    for row in stages:
        print(f"{row['stage']:62s} {row['mean_ms']:9.2f} {row['p95_ms']:9.2f}")
    if args.http:
        report["http_load"] = http_load(args.http.rstrip("/"), args.n, args.workers)
        print("HTTP:", json.dumps(report["http_load"]))
        if args.cleanup:
            report["cleanup_deleted_rows"] = cleanup()
            print("cleanup: deleted", report["cleanup_deleted_rows"], "LOADTEST_ rows")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
