"""Optional live mutation/conflict checks on temporary copies, never the source data."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import KnowledgeAgent
from config import ROOT, Settings
from providers import GeminiProvider
from rag import RAG


def main():
    settings = Settings.from_env()
    provider = GeminiProvider(os.getenv("GEMINI_API_KEY", ""), settings)
    records = []
    with tempfile.TemporaryDirectory(prefix="live-challenge-") as folder:
        data = Path(folder) / "data"
        shutil.copytree(settings.data_dir, data)
        sandbox = replace(settings, data_dir=data)
        old = RAG(sandbox, provider)
        old.ensure_index()
        policy = data / "03_policy_current_v3.txt"
        policy.write_text(policy.read_text().replace("1.1 Customers may return or exchange an unused item within 7 days",
                                                    "1.1 Customers may return or exchange an unused item within 9 days"))
        updated = RAG(sandbox, provider)
        answer = KnowledgeAgent(updated, provider).ask("What is the current return window for an unused ready-made item?")
        ok = updated.fingerprint != old.fingerprint and "9 days" in answer.answer
        records.append({"name": "live_policy_mutation", "pass": ok, "old_fingerprint": old.fingerprint,
                        "new_fingerprint": updated.fingerprint, "actual": answer.to_dict(),
                        "note": "Only a temporary copy changed from 7 to 9 days. Production source files were preserved."})
    with tempfile.TemporaryDirectory(prefix="conflict-check-") as folder:
        data = Path(folder) / "data"
        data.mkdir()
        for filename, days in [("policy_a.txt", 7), ("policy_b.txt", 9)]:
            (data / filename).write_text(f"""---
document: Store Returns Policy
doc_type: policy
policy_version: 3.0
effective_date: 2026-09-01
status: current
---

## Returns
Unused ready-made items may be returned within {days} days of purchase.
This current policy has the same authority as the other current returns policy.
No document establishes which of the two contradictory windows has priority.
""")
        conflict_rag = RAG(replace(settings, data_dir=data), provider)
        answer = KnowledgeAgent(conflict_rag, provider).ask("What is the current return window?", k=5)
        sources = {s["filename"] for s in answer.sources}
        records.append({"name": "unresolved_same_scope_conflict", "pass": answer.status == "conflict" and sources == {"policy_a.txt", "policy_b.txt"},
                        "actual": answer.to_dict(), "note": "Synthetic, isolated test policies with equal version/date/authority."})
    out = ROOT / "evaluation/supplemental_checks.json"
    out.write_text(json.dumps(records, indent=2))
    print(json.dumps([{k: row[k] for k in ["name", "pass"]} for row in records], indent=2))
    if not all(r["pass"] for r in records):
        raise SystemExit("A live supplemental check failed. Review the actual evidence.")


if __name__ == "__main__":
    main()
