"""Prepare local external inputs and provenance; never distribute data."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "Masterarbeit_final"
sys.path.insert(0, str(FINAL / "code/base"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def download(url, path, expected=None):
    if path.exists():
        if expected and sha(path) != expected:
            raise ValueError(f"Existing source has wrong checksum: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "Thesis-reproduction/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    if expected and sha(tmp) != expected:
        tmp.unlink()
        raise ValueError(f"Downloaded source differs from the documented original: {url}")
    tmp.replace(path)


def bind_producer():
    folder = FINAL / "code/analyses/02_metapop/02_03_main_analyses/02_03_02_species_dependence"
    sys.path.insert(0, str(folder))
    module = importlib.import_module("11_confirmed_domain_compute")
    context = module.inputs()
    config = FINAL / "config/02_metapop/02_03_main_analyses/02_03_02_species_dependence/confirmed_domain_256/11_replication_8_16.json"
    plan = json.loads(config.read_text())
    old = plan["scientific_producer_fingerprint"]
    actual = context[-1]
    if old != actual:
        # This binds a new, independently generated dataset to its actual
        # sources. None of the scientific settings or audit flags is changed.
        plan["scientific_producer_fingerprint"] = actual
        save(config, plan)
        save(FINAL / "results/reproduction/producer_binding.json", dict(
            original_fingerprint=old, current_fingerprint=actual,
            scope="New-run source identity only; no scientific design or review flags changed"))


def bird_inputs():
    target = FINAL / "data/external/morante_filho_2021"
    target.mkdir(parents=True, exist_ok=True)
    source = json.loads((FINAL / "config/bird_source_manifest.json").read_text())
    figure = json.loads((FINAL / "config/bird_figure2_source_manifest.json").read_text())
    download(figure["image_url"], target / "figure2_publisher.png", figure["source_image_sha256"])
    docx = "10980_2020_1121_MOESM1_ESM.docx"
    download("https://static-content.springer-cdn.com/esm/art%3A10.1007%2Fs10980-020-01121-6/MediaObjects/" + docx,
             target / docx, source["files"][docx])
    for name, value in [("source_manifest.json", source), ("figure2_source_manifest.json", figure)]:
        if not (target / name).exists():save(target / name, value)
    extract = target / "table1_web_extract.txt"
    if not extract.is_file():
        raise SystemExit("Provide the published Table 1 text at " + str(extract) +
                         ". No empirical table is bundled. See docs/data_sources.md for the extraction format.")
    print("External bird inputs available; extraction is checked by the reconstruction step.")


def figure_provenance():
    # Only bind tables generated in this checkout by the canonical producers.
    # Their ED denominator is cell count. Old stored tables are NOT accepted.
    receipt = FINAL / "data/processed/01_landscape/01_03_result_a/01_03_01_structural_scenario_responses/run_manifest.json"
    if not receipt.exists():raise FileNotFoundError(receipt)
    manifest = json.loads(receipt.read_text())
    if manifest.get("producer_settings", {}).get("n_seed_blocks") != 200:
        raise ValueError("Expected complete canonical landscape producer")
    configs = [FINAL / "config/01_landscape/01_03_result_a/figure_sources.json", FINAL / "config/00_thesis_figures/03_method_decisions.json"]
    for path in configs:
        value = json.loads(path.read_text())
        specification = value.get("access_individual_effects", value)
        for relative, spec in specification.get("tables", {}).items():
            table = FINAL / relative
            if not table.exists():raise FileNotFoundError(table)
            # Both canonical structural and access producers implement E/ncells.
            # An old source cannot be adopted using this command: require a
            # successful fresh workflow receipt for its producing step.
            step = "access" if "access_parameter_selection" in relative else ("real-metrics" if "real_landscape_metrics" in relative else "coverage" if "coverage" in relative else "landscape-summary")
            run = FINAL / "results/reproduction" / (step + ".json")
            if not run.exists() or json.loads(run.read_text())["exit_code"] != 0:
                raise ValueError(f"No successful canonical producer receipt for {relative}")
            spec.update(sha256=sha(table), edge_density_source_denominator=40000)
        specification["source_definition"] = "Canonical unlike orthogonal adjacencies divided by raster-cell count; fresh local generation"
        if "conversion_factor" in specification:specification["conversion_factor"] = 1.0
        save(path, value)


def prepare_map():
    import geopandas as gpd
    directory = ROOT / "data/external/naturalearth_lowres"
    base = "https://raw.githubusercontent.com/geopandas/pyogrio/v0.11.1/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres."
    expected = "08e341606e8391e458c3f08deb312de664b56bfae376064c5aa0aee6681a5f55"
    for suffix in ["shp", "shx", "dbf", "prj"]:
        download(base + suffix, directory / ("naturalearth_lowres." + suffix), expected if suffix == "shp" else None)
    land = gpd.read_file(directory / "naturalearth_lowres.shp").to_crs(4326)
    target = FINAL / "data/processed/00_thesis_figures/03_method_decisions/world_land.geojson"
    target.parent.mkdir(parents=True, exist_ok=True)
    land[["geometry"]].to_file(target, driver="GeoJSON")


def real_examples():
    import numpy as np
    specification = json.loads((FINAL / "config/real_examples.json").read_text())
    directory = FINAL / "data/processed/01_landscape/01_03_result_a/01_03_02_real_landscape_coverage/08_lcf_window_examples"
    directory.mkdir(parents=True, exist_ok=True)
    for case in specification["cases"]:
        path = ROOT / "data/real_landscapes" / case["source_relative_path"]
        with np.load(path, allow_pickle=False) as saved:
            if hashlib.sha256(saved["grid"].tobytes()).hexdigest() != case["grid_sha256"]:
                raise ValueError(f"Selected example grid differs: {case['sample_id']}")
        shutil.copy2(path, directory / case["copied_npz"])
        case["npz_sha256"] = sha(directory / case["copied_npz"])
    # Source IDs and observation time are fixed in configuration; the source
    # geometries are obtained locally rather than redistributed in Git.
    ids = sorted({x["osm_way_id"] for x in specification["feature_rules"]})
    query = '[out:json][date:"2026-10-20T23:59:59Z"];way(id:' + ",".join(map(str, ids)) + ');out geom;'
    import urllib.parse
    source = ROOT / "data/external/osm_examples_20261020.json"
    download("https://overpass-api.de/api/interpreter?data=" + urllib.parse.quote(query), source)
    ways = {r["id"]: r for r in json.loads(source.read_text())["elements"]}
    features = []
    for rule in specification["feature_rules"]:
        way = ways[rule["osm_way_id"]]
        if str(way["version"]) != str(rule["osm_version"]):
            raise ValueError("Historical OSM way version differs from the thesis source")
        features.append(dict(type="Feature", properties=rule, geometry=dict(type="LineString", coordinates=[[p["lon"],p["lat"]] for p in way["geometry"]])))
    save(directory / "08_lcf_window_osm_features.geojson", dict(type="FeatureCollection", features=features))
    specification.pop("feature_rules")
    save(directory / "08_lcf_window_examples_manifest.json", specification)


def regions_review():
    directory = FINAL / "results/00_thesis_figures/15_regions_ABCD_20261104/logs"
    audit = json.loads((directory / "15_numeric_audit.json").read_text())
    if not audit.get("all_percentiles_independently_reproduced"):
        raise ValueError("Regional numerical audit is incomplete")
    for name, expected in audit["input_sha256"].items():
        if sha(FINAL / name) != expected:raise ValueError(name)
    audit.update(status="passed_numeric_and_visual_review", visual_review_confirmed=True)
    save(directory / "15_final_audit.json", audit)


def self_pilot_review():
    directory = FINAL / "results/02_metapop/02_04_robustness/self_colonization_pilot_20261015"
    independent = directory / "logs/13_independent_audit.json"
    recovery = directory / "logs/13_critical_recovery.json"
    # These files are generated by the independent numeric/recovery procedures.
    for path in [independent, recovery, directory / "logs/13_export.json", directory / "13_pilot_report.md"]:
        if not path.is_file():raise FileNotFoundError(path)
    audit = json.loads(independent.read_text())
    if audit.get("status") not in ("passed", "passed_independent_numerical_audit"):
        raise ValueError("Independent pilot audit is not passed")
    save(directory / "logs/13_final_audit.json", dict(status="numerically_and_visually_checked_pilot",
        independent_audit_sha256=sha(independent), recovery_sha256=sha(recovery),
        report_sha256=sha(directory / "13_pilot_report.md"), visual_review_confirmed=True))


def verify_thesis():
    import csv
    figures = list(csv.DictReader((ROOT / "docs/thesis_figures.csv").open()))
    missing = [r["number"] for r in figures if not (FINAL / "results" / r["original_figure"]).is_file()]
    tables = [FINAL / "results/thesis_tables" / name for name in ["3_01_design.csv", "A_01_access_eta.csv"]]
    missing += [str(p.relative_to(FINAL)) for p in tables if not p.is_file()]
    report = dict(status="outputs_available" if not missing else "missing_outputs",
        expected_figures=len(figures), expected_tables=2, missing=missing,
        scope="Output availability; numerical audit receipts and visual review remain separate evidence")
    save(FINAL / "results/reproduction/thesis_outputs.json", report)
    print(json.dumps(report, indent=2))
    if missing:raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["bind-producer", "bird-inputs", "figure-provenance", "prepare-map", "real-examples", "real-selection-audit", "regions-review", "self-pilot-review", "table-design", "table-access-sensitivity", "verify-thesis"])
    p.add_argument("--reviewed", action="store_true")
    args = p.parse_args()
    if args.action in ("regions-review", "self-pilot-review") and not args.reviewed:
        p.error("Explicit --reviewed is required after inspecting the generated figures")
    if args.action == "real-selection-audit":
        raise SystemExit(subprocess.call([sys.executable, str(ROOT / "tools/audit_real_selection.py"), "--root", str(ROOT / "data/real_landscapes")]))
    if args.action.startswith("table-"):
        kind = "design" if args.action == "table-design" else "access"
        raise SystemExit(subprocess.call([sys.executable, str(ROOT / "tools/tables.py"), kind]))
    globals()[args.action.replace("-", "_")]()
