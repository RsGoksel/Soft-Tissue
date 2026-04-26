"""
Poll the webhook.site bucket and persist every Q&A entry to a local
JSONL file so the operator has durable on-disk records.

Reads the webhook URL from `.webhook_url` (written by
`generate_hoca_dashboard.py`). Polls every 8 seconds. New entries are
appended to `outputs/qa_log/qa_log.jsonl`; duplicates are skipped via
the webhook.site request UUID.

Run this once in a terminal and leave it open during the advisor's
session. Quit with Ctrl+C.

Usage:
    python scripts/collect_qa_log.py
    python scripts/collect_qa_log.py --interval 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEBHOOK_PATH = ROOT / ".webhook_url"
OUT_PATH = ROOT / "outputs" / "qa_log" / "qa_log.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=8,
                   help="seconds between polls (default 8)")
    p.add_argument("--per-page", type=int, default=50,
                   help="webhook.site page size (default 50)")
    return p.parse_args()


def webhook_uuid() -> str:
    if not WEBHOOK_PATH.exists():
        sys.exit(f"error: {WEBHOOK_PATH} not found — run "
                 f"scripts/generate_hoca_dashboard.py first")
    url = WEBHOOK_PATH.read_text(encoding="utf-8").strip()
    if not url:
        sys.exit("error: .webhook_url is empty")
    return url.rstrip("/").split("/")[-1]


def load_seen_ids() -> set[str]:
    seen = set()
    if not OUT_PATH.exists():
        return seen
    for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
            wid = e.get("_webhook_id")
            if wid:
                seen.add(wid)
        except json.JSONDecodeError:
            pass
    return seen


def fetch_requests(uuid: str, per_page: int) -> list[dict]:
    url = (f"https://webhook.site/token/{uuid}/requests"
           f"?sorting=newest&per_page={per_page}")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        body = r.read().decode("utf-8")
    return json.loads(body).get("data", [])


def parse_payload(req_obj: dict) -> dict:
    raw = req_obj.get("content", "")
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {"_raw": raw[:1000]}
    payload["_webhook_id"]   = req_obj.get("uuid")
    payload["_received_at"]  = req_obj.get("created_at")
    payload["_remote_ip"]    = req_obj.get("ip")
    return payload


def write_entry(entry: dict) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def short(s: str, n: int = 80) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def main() -> None:
    args = parse_args()
    uuid = webhook_uuid()
    print(f"[collector] webhook uuid: {uuid}")
    print(f"[collector] view URL: https://webhook.site/#!/view/{uuid}")
    print(f"[collector] writing -> {OUT_PATH.relative_to(ROOT)}")
    print(f"[collector] poll every {args.interval}s — Ctrl+C to stop\n")

    seen = load_seen_ids()
    print(f"[collector] {len(seen)} entries already on disk\n")

    fail_streak = 0
    while True:
        try:
            requests = fetch_requests(uuid, args.per_page)
            fail_streak = 0
            new = []
            # webhook.site returns newest first; flip to chronological
            for req in reversed(requests):
                if req.get("uuid") in seen:
                    continue
                payload = parse_payload(req)
                seen.add(payload["_webhook_id"])
                write_entry(payload)
                new.append(payload)

            if new:
                stamp = datetime.now().strftime("%H:%M:%S")
                for p in new:
                    role = p.get("role", "?")
                    body = short(p.get("content", ""))
                    extra = ""
                    if role == "assistant":
                        ms = p.get("latency_ms")
                        extra = f"  ({ms} ms)" if ms else ""
                    print(f"[{stamp}] [{role:>9}]{extra}  {body}")
        except urllib.error.URLError as exc:
            fail_streak += 1
            print(f"[collector] network error ({fail_streak}): {exc}",
                  file=sys.stderr)
            if fail_streak > 5:
                print("[collector] too many failures, giving up",
                      file=sys.stderr)
                sys.exit(1)
        except KeyboardInterrupt:
            print("\n[collector] stopped by user")
            return
        except Exception as exc:
            print(f"[collector] unexpected error: {exc}", file=sys.stderr)

        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[collector] stopped by user")
            return


if __name__ == "__main__":
    main()
