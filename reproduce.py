"""Data-free entry point for the final thesis supplement (Python 3.12)."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "Masterarbeit_final"
BASE = FINAL / "code/base"
REGISTRY = ROOT / "workflow.json"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache/numba"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["PYTHONPATH"] = os.pathsep.join([str(BASE), str(ROOT), os.environ.get("PYTHONPATH", "")])
sys.path.insert(0, str(BASE))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry():
    return json.loads(REGISTRY.read_text())["steps"]


def ordered(targets, steps):
    result, active, done = [], set(), set()

    def visit(name):
        if name in active:
            raise ValueError(f"Workflow cycle: {name}")
        if name in done:
            return
        if name not in steps:
            raise ValueError(f"Unregistered prerequisite: {name}")
        active.add(name)
        for dependency in steps[name].get("needs", []):
            visit(dependency)
        active.remove(name)
        done.add(name)
        result.append(name)

    for name in targets:
        visit(name)
    return result


def command(step):
    if "module" in step:
        cmd = [sys.executable, "-m", step["module"]]
    elif "action" in step:
        cmd = [sys.executable, str(ROOT / "tools/prepare.py"), step["action"]]
    else:
        cmd = [sys.executable, str(ROOT / step["script"])]
    return cmd + [str(x).replace("{root}", str(ROOT)) for x in step.get("args", [])]


def check():
    steps = registry()
    ordered(list(steps), steps)
    errors = []
    for name, step in steps.items():
        if "script" in step and not (ROOT / step["script"]).is_file():
            errors.append(f"{name}: missing script {step['script']}")
    for path in sorted((FINAL / "code").rglob("*.py")):
        try:
            ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            errors.append(str(exc))
    figures = list(csv.DictReader((ROOT / "docs/thesis_figures.csv").open()))
    for row in figures:
        path = ROOT / row["reference_file"]
        if not path.exists() or sha(path) != row["figure_sha256"]:
            errors.append(f"Missing or changed final reference figure: {row['id']}")
        if row.get("step") not in steps:
            errors.append(f"Figure without registered step: {row['id']}")
    promises = list(csv.DictReader((ROOT / "docs/repository_promises.csv").open()))
    for row in promises:
        path, _, anchor = row["documentation"].partition("#")
        doc = ROOT / path
        if not doc.is_file() or (anchor and f'id="{anchor}"' not in doc.read_text()):
            errors.append(f"Missing promised documentation: {row['id']}")
    imported = {}
    import importlib.metadata
    for package in ("numpy", "pandas", "scipy", "matplotlib", "rasterio", "shapely", "numba", "pillow", "pyproj", "geopandas", "pyogrio", "pyshp"):
        try:
            imported[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing dependency: {package}")
    report = dict(status="passed" if not errors else "failed", registered_steps=len(steps),
                  final_figures=len(figures), manuscript_promises=len(promises),
                  python=sys.version.split()[0], packages=imported, errors=errors,
                  scope="Source/registry/reference integrity only; not production recomputation")
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


def run_step(name, extra, review):
    steps = registry()
    if name not in steps:
        raise SystemExit(f"Unknown step {name!r}; use list")
    step = steps[name]
    if step.get("requires_review") and not review:
        raise SystemExit(f"{name} records an actual visual review. Inspect the preceding outputs, then use --reviewed. See docs/reproducibility.md.")
    for raw in step.get("inputs", []):
        p = ROOT / raw
        if not p.exists():
            raise SystemExit(f"{name}: missing input {raw}. Run its producer or acquire the documented external source; see docs/data_sources.md.")
    cmd = command(step) + extra
    started = time.time()
    print(f"Running {name}", flush=True)
    code = subprocess.call(cmd, cwd=ROOT)
    log = FINAL / "results/reproduction"
    log.mkdir(parents=True, exist_ok=True)
    fingerprint = {str(p.relative_to(ROOT)): sha(p) for p in [REGISTRY, ROOT / "reproduce.py"]}
    (log / f"{name}.json").write_text(json.dumps(dict(step=name, command=cmd, exit_code=code,
        elapsed_seconds=time.time()-started, entry_sha256=fingerprint,
        scope="Successful command exit only; scientific audits remain separate"), indent=2) + "\n")
    if code:
        raise SystemExit(code)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=["check", "test", "list", "plan", "run", "pipeline"])
    p.add_argument("step", nargs="?")
    p.add_argument("--reviewed", action="store_true", help="Only after actually reviewing outputs")
    args, extra = p.parse_known_args()
    if extra and args.operation != "run":
        p.error("Additional script arguments are only accepted by run")
    if args.operation == "check":
        raise SystemExit(check())
    if args.operation == "test":
        import unittest
        loader = unittest.TestLoader()
        suite = unittest.TestSuite([unittest.TestLoader().discover(str(FINAL / "tests/base")),
                                  unittest.TestLoader().discover(str(FINAL / "tests"))])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(not result.wasSuccessful())
    steps = registry()
    if args.operation == "list":
        for key, step in steps.items():
            print(f"{key:34s} {step['description']}")
        return
    if args.operation == "plan":
        for key in ordered([args.step] if args.step else list(steps), steps):
            step = steps[key]
            print(f"{key}: {step['description']}")
            print("  " + " ".join(command(step)))
            if step.get("requires_review"):
                print("  Requires actual visual review before confirming.")
        return
    if not args.step:
        p.error("run and pipeline require a step")
    for key in ordered([args.step], steps) if args.operation == "pipeline" else [args.step]:
        run_step(key, extra, args.reviewed if args.operation == "run" else False)


if __name__ == "__main__":
    main()
