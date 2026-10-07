"""End-to-end check / demo seeder for a RUNNING TrustX stack.

Walks the investigation workflow against the real API (and, if the frontend is
running, through its server-side proxy):

    analyze a normal and a PaySim-style suspicious transaction -> model results ->
    network/behavioral signals -> counterfactual -> investigation -> analyst review
    create/update -> audit log -> review queue

It prints what the model ACTUALLY returned. Nothing is assumed about risk levels.
Rows created here are ordinary stored analyses (duplicate protection makes re-runs
idempotent); there is no API to delete them.

Usage (from the repository root, with the backend running):

    backend\\.venv\\Scripts\\python.exe scripts\\e2e_demo.py
    backend\\.venv\\Scripts\\python.exe scripts\\e2e_demo.py --api http://127.0.0.1:8000 --frontend http://127.0.0.1:3000

API keys are read from the repository-root .env (TRUSTX_ANALYST_API_KEY /
TRUSTX_ADMIN_API_KEY) or from the environment. They are never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, bool, str]] = []


def load_env() -> None:
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def call(base: str, method: str, path: str, key: str | None = None, body=None, user: str | None = None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-API-Key"] = key
    if user:
        headers["X-TrustX-User"] = user
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            try:
                return response.status, json.loads(raw or b"null")
            except json.JSONDecodeError:
                return response.status, raw.decode(errors="replace")
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, raw.decode(errors="replace")


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def tx(step, kind, amount, sender, old_o, new_o, recipient, old_d, new_d, history=None):
    return {
        "step": step, "type": kind, "amount": amount, "nameOrig": sender,
        "oldbalanceOrg": old_o, "newbalanceOrig": new_o, "nameDest": recipient,
        "oldbalanceDest": old_d, "newbalanceDest": new_d,
        "historical_transactions": history or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend", default=None, help="e.g. http://127.0.0.1:3000")
    args = parser.parse_args()
    load_env()
    analyst, admin = os.getenv("TRUSTX_ANALYST_API_KEY"), os.getenv("TRUSTX_ADMIN_API_KEY")
    if not analyst or not admin:
        print("TRUSTX_ANALYST_API_KEY / TRUSTX_ADMIN_API_KEY are not configured.")
        return 2
    api = args.api.rstrip("/")

    status, body = call(api, "GET", "/health")
    check("health is public", status == 200 and body.get("status") == "ok")
    check("analyze without key -> 401", call(api, "POST", "/api/v1/transactions/analyze", None, {})[0] == 401)
    check("analyst blocked from model status -> 403", call(api, "GET", "/api/v1/models/status", analyst)[0] == 403)
    status, body = call(api, "GET", "/api/v1/models/status", admin)
    check("admin can read model status", status == 200 and body["metadata"]["feature_count"] == 28)

    # 1. A normal PaySim PAYMENT row and a suspicious PaySim-style TRANSFER row.
    normal_payload = tx(1, "PAYMENT", 1864.28, "C1666544295", 21249.0, 19384.72, "M2044282225", 0.0, 0.0)
    suspicious_payload = tx(1, "TRANSFER", 181.0, "C1305486145", 181.0, 0.0, "C553264065", 0.0, 0.0)
    results = {}
    for label, payload in (("normal", normal_payload), ("suspicious", suspicious_payload)):
        status, analysis = call(api, "POST", "/api/v1/transactions/analyze", analyst, payload)
        check(f"analyze {label} transaction", status == 200, f"status {status}")
        if status != 200:
            print(analysis)
            return 1
        results[label] = analysis
        print(
            f"      -> {label}: model score {analysis['fraud_probability']:.6f}, "
            f"risk {analysis['risk_level']}, action {analysis['recommended_action']}, "
            f"anomaly {analysis['anomaly_signal']:.4f}, duplicate={analysis['is_duplicate']}"
        )

    suspicious = results["suspicious"]
    tid = suspicious["transaction_id"]
    check("SHAP explanation available", suspicious["explanation"]["status"] == "available")
    check("evidence returned", len(suspicious["evidence"]["items"]) >= 3)
    check("investigator summary present", bool(suspicious["investigator_summary"]))
    check("summary does not assert fraud",
          "definitely" not in suspicious["investigator_summary"]["investigator_summary"].lower())

    # 2. Duplicate protection.
    status, again = call(api, "POST", "/api/v1/transactions/analyze", analyst, suspicious_payload)
    check("resubmission returns stored analysis", again.get("is_duplicate") is True and again["transaction_id"] == tid)

    # 3. Network fan-in: several distinct senders to one recipient at increasing steps.
    recipient = "DEMO_FANIN_RECIPIENT"
    last = None
    for index in range(1, 7):
        payload = tx(10 + index, "TRANSFER", 500.0 + index, f"DEMO_SENDER_{index}", 5000.0, 4500.0 - index,
                     recipient, 100.0, 600.0 + index)
        status, last = call(api, "POST", "/api/v1/transactions/analyze", analyst, payload)
    network = last["network_signals"]
    patterns = [p["pattern"] for p in network["patterns"] if p["detected"]]
    check("network signals persisted at analysis time", network["source"] == "analysis_time")
    check("high recipient connectivity observed after 6 distinct senders",
          "HIGH_RECIPIENT_CONNECTIVITY" in patterns, f"unique senders={network['recipient']['unique_sender_count']}")

    # 4. Investigation views.
    status, detail = call(api, "GET", f"/api/v1/transactions/{tid}", analyst)
    check("stored investigation retrievable", status == 200 and detail["transaction_id"] == tid)
    status, net = call(api, "GET", f"/api/v1/transactions/{tid}/network", analyst)
    check("network endpoint returns stored result", status == 200 and net["source"] == "analysis_time")
    status, behavior = call(api, "GET", f"/api/v1/transactions/{tid}/behavior", analyst)
    check("behavioral endpoint works", status == 200 and behavior["status"] == "available")
    status, cf = call(api, "GET", f"/api/v1/transactions/{tid}/counterfactual", analyst)
    check("counterfactual works", status == 200 and len(cf["counterfactuals"]) == 4)
    if status == 200:
        print("      -> counterfactual scenarios (ratio: valid, score, change vs baseline):")
        for item in cf["counterfactuals"]:
            print(f"         {item['amount_ratio']:.2f}: valid={item['valid']}, score={item['fraud_probability']}, change={item['probability_change']}")
        print(f"      -> baseline source={cf['baseline_source']}, matches stored={cf['baseline_matches_stored']}, best={bool(cf['best_counterfactual'])}")

    # 5. Analyst review workflow + audit.
    status, review = call(api, "POST", f"/api/v1/reviews/{tid}", analyst,
                          {"status": "UNDER_REVIEW", "analyst_note": "Demo: contacting sender."}, user="demo_analyst")
    if status == 409:
        status, review = call(api, "GET", f"/api/v1/reviews/{tid}", analyst)
    check("review created (or already exists)", status in (200, 201), f"status {status}")
    status, review = call(api, "PATCH", f"/api/v1/reviews/{tid}", analyst,
                          {"status": "ESCALATED", "decision": "Escalate for senior review"}, user="demo_analyst")
    check("review updated", status == 200 and review["status"] == "ESCALATED" and review["reviewed_by"] == "analyst:demo_analyst")
    status, after = call(api, "GET", f"/api/v1/transactions/{tid}", analyst)
    check("model output unchanged by review",
          after["risk_level"] == suspicious["risk_level"] and after["fraud_probability"] == suspicious["fraud_probability"])
    status, audit = call(api, "GET", f"/api/v1/audit-logs?resource_id={tid}&page_size=50", admin)
    actions = {entry["action"] for entry in audit["items"]} if status == 200 else set()
    check("audit log contains analyze/view/review events",
          {"TRANSACTION_ANALYZE", "TRANSACTION_VIEW", "REVIEW_CREATE", "REVIEW_UPDATE"} <= actions, ", ".join(sorted(actions)))

    # 6. Review queue + dashboard summary.
    status, queue = call(api, "GET", "/api/v1/transactions?review_queue=true&sort_by=risk_priority&page_size=100", analyst)
    queue_ids = [item["transaction_id"] for item in queue["items"]] if status == 200 else []
    check("review queue lists only HIGH/CRITICAL", status == 200 and all(i["risk_level"] in ("HIGH", "CRITICAL") for i in queue["items"]))
    if suspicious["risk_level"] in ("HIGH", "CRITICAL"):
        check("escalated item stays in the queue until a reviewer closes it", tid in queue_ids)
        call(api, "PATCH", f"/api/v1/reviews/{tid}", analyst, {"status": "DISMISSED"}, user="demo_analyst")
        status, queue = call(api, "GET", "/api/v1/transactions?review_queue=true&page_size=100", analyst)
        check("dismissed item leaves the queue", tid not in [i["transaction_id"] for i in queue["items"]])
        call(api, "PATCH", f"/api/v1/reviews/{tid}", analyst, {"status": "UNDER_REVIEW"}, user="demo_analyst")
    else:
        print("      (the model did not rate the suspicious example HIGH/CRITICAL; queue checks skipped)")
    status, summary = call(api, "GET", "/api/v1/transactions/summary", analyst)
    print("      -> risk distribution in DB:", summary["count_by_risk_level"])

    # 7. Frontend proxy.
    if args.frontend:
        front = args.frontend.rstrip("/")
        check("frontend home loads", call(front, "GET", "/")[0] == 200)
        status, body = call(front, "GET", "/api/trustx/api/v1/transactions/summary")
        check("frontend proxy reaches the API with the server-side key", status == 200 and "total_analyzed_transactions" in body)
        check("frontend proxy blocks admin endpoints", call(front, "GET", "/api/trustx/api/v1/models/status")[0] == 403)
        check("frontend proxy blocks audit log", call(front, "GET", "/api/trustx/api/v1/audit-logs")[0] == 403)
        check("investigation page loads", call(front, "GET", f"/transactions/{tid}")[0] == 200)
        check("analyze page loads", call(front, "GET", "/analyze")[0] == 200)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name in failed:
        print("FAILED:", name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
