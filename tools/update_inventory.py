"""Record the exact tracked source bundle without adding local generated data."""
import csv
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "docs/source_inventory.csv"
old = {row["path"]: row for row in csv.DictReader(path.open())} if path.exists() else {}
tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
import json
summary_path = ROOT / "docs/build_summary.json"
summary = json.loads(summary_path.read_text())
summary.pop("source_files", None)
summary["tracked_file_count"] = sum(bool(x) for x in tracked)
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
rows = []
for name in sorted(x for x in tracked if x and x != "docs/source_inventory.csv"):
    source = ROOT / name
    if not source.is_file():continue
    prior = old.get(name, {})
    rows.append(dict(path=name, source=prior.get("source", "repository implementation or compiled documentation"),
        original_sha256=prior.get("original_sha256", prior.get("sha256", "")),
        export_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), bytes=source.stat().st_size))
with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader();writer.writerows(rows)
print(f"Inventoried {len(rows)} tracked files, {sum(r['bytes'] for r in rows)/1048576:.2f} MiB; inventory excludes itself")
