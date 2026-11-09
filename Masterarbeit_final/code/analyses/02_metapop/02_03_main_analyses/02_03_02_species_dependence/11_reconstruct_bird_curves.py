"""Reconstruct published bird response curves from rounded Table 1 predictions.

The coefficients are empirical logit-curve approximations, not metapopulation
parameters. No statistical uncertainty is inferred from rounded predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[6]
DEFAULT_CONFIG = ROOT / "Masterarbeit_final/config/02_metapop/02_03_main_analyses/02_03_02_species_dependence/bird_reconstruction/11_bird_reconstruction.json"
DOI = "https://doi.org/10.1007/s10980-020-01121-6"
SOURCE_URL = "https://www.researchgate.net/publication/344284648_Landscape_composition_is_the_strongest_determinant_of_bird_occupancy_patterns_in_tropical_forest_patches"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logit(p: float) -> float:
    if not 0 < p < 1:
        raise ValueError(f"Finite logit requires 0 < p < 1, received {p}")
    return math.log(p) - math.log1p(-p)


def expit(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    y = math.exp(x)
    return y / (1 + y)


def coefficients(p20: float, p40: float) -> tuple[float, float]:
    if p40 <= p20:
        raise ValueError("Selected forest curves must increase between the anchors")
    b1 = (logit(p40) - logit(p20)) / 0.2
    return logit(p20) - b1 * 0.2, b1


def threshold(p20: float, p40: float, probability: float = 0.6) -> float:
    b0, b1 = coefficients(p20, p40)
    return (logit(probability) - b0) / b1


def rounding_threshold_bounds(p20: float, p40: float, half_width: float) -> tuple[float, float]:
    """Extrema over the rounding rectangle; not sampling uncertainty.

    In logit coordinates the threshold is a ratio of affine functions with a
    strictly positive denominator. Its extrema on this rectangle occur at
    vertices. Reject rectangles touching probability bounds or zero slope.
    """
    if not (0 < p20 - half_width < p20 + half_width < p40 - half_width
            < p40 + half_width < 1):
        raise ValueError("Rounding rectangle must have finite logits and a positive slope")
    values = [threshold(a, b) for a, b in itertools.product(
        (p20 - half_width, p20 + half_width),
        (p40 - half_width, p40 + half_width))]
    return min(values), max(values)


def source_rows(path: Path) -> list[dict]:
    rows = []
    pattern = re.compile(r"^L\d+: (.*?) (Non-forest|Forest) (frugivore|insectivore|omnivore|nectarivore) (0\.\d+|1\.00) (0\.\d+|1\.00) (\d+\.\d+)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.fullmatch(line)
        if not match:
            continue
        species, habitat, guild, p20, p40, h60 = match.groups()
        index = len(rows) + 1
        rows.append({
            "species_id": f"mf2021_{index:02d}",
            "table1_row": index,
            "species_as_published": unicodedata.normalize("NFKC", species),
            "habitat_specialization": "forest_dependent" if habitat == "Forest" else "non_forest_dependent",
            "trophic_guild": guild,
            "published_psi_H020": p20,
            "published_psi_H040": p40,
            "published_H_psi060_percent": h60,
            "included_in_reconstruction": habitat == "Forest",
        })
    return rows


def csv_bytes(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(csv_bytes(rows))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def supplement_classification(path: Path) -> dict[str, tuple[str, str]]:
    with ZipFile(path) as archive:
        doc = ET.fromstring(archive.read("word/document.xml"))
    tables = doc.findall(".//w:tbl", NS)
    rows = []
    for table in tables:
        cells = [["".join(t.text or "" for t in cell.findall(".//w:t", NS))
                  for cell in row.findall("w:tc", NS)]
                 for row in table.findall("w:tr", NS)]
        if cells and cells[0][:2] == ["Species", "Habitat specialization"]:
            rows = cells[1:]
            break
    if not rows:
        raise ValueError("Supplement Table S2 was not found")
    return {r[0].strip(): (r[1].strip(), r[2].strip().lower()) for r in rows}


def run(config_path: Path, output_override: Path | None = None) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    cfg = json.loads(config_path.read_text())
    if cfg["anchor_h"] != [0.2, 0.4] or cfg["check_occupancy"] != 0.6:
        raise ValueError("This entry point implements the frozen Table 1 anchors and check")
    source_dir = ROOT / cfg["source_directory"]
    output_dir = output_override or ROOT / cfg["output_directory"]
    audit_dir = output_dir / "logs" if output_override else ROOT / cfg["audit_directory"]
    extract = source_dir / "table1_web_extract.txt"
    supplement = source_dir / cfg["supplement_filename"]
    grid_config = ROOT / cfg["simulation_grid_config"]
    rows = source_rows(extract)
    selected = [r for r in rows if r["included_in_reconstruction"]]
    assert len(rows) == cfg["expected_table1_species"] == 19
    assert len(selected) == cfg["expected_forest_species"] == 15
    assert len({r["species_id"] for r in rows}) == len({r["species_as_published"] for r in rows}) == 19
    assert all(0 <= float(r[k]) <= 1 for r in rows for k in ("published_psi_H020", "published_psi_H040"))

    # Preserve extracted spelling; this explicit alias is solely a source join.
    name_alias = {"Drimophila squamata": "Drymophilla squamata"}
    classification = supplement_classification(supplement)
    for r in rows:
        name = name_alias.get(r["species_as_published"], r["species_as_published"])
        habitat, guild = classification[name]
        assert (habitat == "Forest dependent") == r["included_in_reconstruction"]
        assert guild == r["trophic_guild"]
    table = source_dir / "table1.csv"
    # Changing an existing source CSV requires an explicit reviewed reimport.
    if table.exists() and table.read_bytes() != csv_bytes(rows):
        raise ValueError("Existing table1.csv differs from retained source extraction")
    if not table.exists():
        table.write_bytes(csv_bytes(rows))

    hmin, hmax = cfg["observed_h_min"], cfg["observed_h_max"]
    assert 0 < hmin < 0.2 < 0.4 < hmax < 1
    step = Decimal(str(cfg["dense_h_step"]))
    dense_h = [float(Decimal(i) * step) for i in range(math.ceil(hmin / float(step)), math.floor(hmax / float(step)) + 1)]
    dense_h = sorted(set([hmin, hmax, *dense_h]))
    all_main_h = json.loads(grid_config.read_text())["h_targets"]
    assert len(all_main_h) == len(set(all_main_h)) == 88
    main_h = [h for h in all_main_h if hmin <= h <= hmax]
    coef_rows, dense_rows, main_rows = [], [], []
    max_anchor_error = max_independent_error = 0.0
    for r in selected:
        p20, p40 = float(r["published_psi_H020"]), float(r["published_psi_H040"])
        b0, b1 = coefficients(p20, p40)
        hp = float(r["published_H_psi060_percent"]) / 100
        hr = threshold(p20, p40)
        lower, upper = rounding_threshold_bounds(p20, p40, cfg["probability_rounding_half_width"])
        compatible = lower <= hp + cfg["threshold_rounding_half_width_h"] and upper >= hp - cfg["threshold_rounding_half_width_h"]
        assert compatible, f"Published threshold is incompatible with rounding: {r['species_as_published']}"
        errors = [abs(expit(b0 + b1 * h) - p) for h, p in ((0.2, p20), (0.4, p40))]
        max_anchor_error = max(max_anchor_error, *errors)
        coef_rows.append({
            "species_id": r["species_id"], "species_as_published": r["species_as_published"],
            "trophic_guild": r["trophic_guild"],
            "beta0_reconstructed_logit": b0, "beta1_reconstructed_per_H": b1,
            "observed_H_min": hmin, "observed_H_max": hmax,
            "published_H_psi060": hp, "reconstructed_H_psi060": hr,
            "H_psi060_difference_percentage_points": 100 * (hr - hp),
            "reconstructed_psi_at_published_H060": expit(b0 + b1 * hp),
            "rounding_H060_lower": lower, "rounding_H060_upper": upper,
            "threshold_compatible_with_rounding": compatible,
            "empirical_ci_available": False,
        })
        odds20 = p20 / (1 - p20)
        odds_ratio = (p40 / (1 - p40)) / odds20
        for hs, target in ((dense_h, dense_rows), (main_h, main_rows)):
            previous = -1.0
            for h in hs:
                psi = expit(b0 + b1 * h)
                # Independent odds interpolation instead of coefficient evaluation.
                odds = odds20 * odds_ratio ** ((h - 0.2) / 0.2)
                independent = odds / (1 + odds)
                max_independent_error = max(max_independent_error, abs(psi - independent))
                assert 0 < psi < 1 and psi >= previous and hmin <= h <= hmax
                previous = psi
                target.append({"species_id": r["species_id"], "species_as_published": r["species_as_published"],
                               "H": h, "psi_reconstructed": psi})
    assert max_anchor_error < 1e-14 and max_independent_error < 1e-14
    assert len(dense_rows) == len(selected) * len(dense_h)
    assert len(main_rows) == len(selected) * len(main_h)
    for sample in (dense_rows, main_rows):
        assert len({(r["species_id"], r["H"]) for r in sample}) == len(sample)

    outputs = {
        "coefficients": output_dir / "11_bird_curve_coefficients.csv",
        "dense_curves": output_dir / "11_bird_curves_dense.csv",
        "main_grid_curves": output_dir / "11_bird_curves_main_H_grid.csv",
    }
    for key, sample in zip(outputs, (coef_rows, dense_rows, main_rows)):
        write_csv(outputs[key], sample)
        with outputs[key].open(newline="", encoding="utf-8") as stream:
            saved = list(csv.DictReader(stream))
        assert len(saved) == len(sample)
        for expected, actual in zip(sample, saved):
            assert all(str(v) == actual[k] for k, v in expected.items())

    source_manifest = {
        "dataset_id": "morante_filho_2021_table1", "schema_version": cfg["schema_version"],
        "citation": "Morante-Filho, J.C., Benchimol, M. & Faria, D. (2021). Landscape Ecology 36, 105-117.",
        "doi": DOI, "source_url": SOURCE_URL, "retrieved_on": "2026-10-13",
        "data_type": "Rounded published model predictions; not detection histories or raw observations",
        "source_table": "Main article Table 1", "source_rows": 19,
        "copyright": "Original article copyright Springer Nature B.V. 2020; no open licence identified",
        "verification": "Numbers parsed directly from retained author full-text table extraction; Table S2 classifications cross-checked; publisher PDF table not visually verified",
        "spelling": "Unicode compatibility ligatures normalised; extracted source species spelling otherwise retained. Drimophila squamata joins supplementary Drymophilla squamata explicitly; verify taxonomy before external joins.",
        "classification_aliases": name_alias,
        "selection": cfg["selection_rule"],
        "selection_qualification": "Table 1 membership is used, rather than a new minimum-AIC filter. For Crypturellus soui, Table S4 lists an understory model at minimum AIC and supported forest-cover models within delta AIC 2.",
        "files": {p.name: sha256(p) for p in (extract, table, supplement)},
        "columns": {
            "species_id": "Stable table-row identifier; not a taxonomic authority identifier",
            "table1_row": "1-based source data-row order",
            "species_as_published": "Name from the retained main-article text extraction",
            "habitat_specialization": "Source Forest or Non-forest classification",
            "trophic_guild": "Source main diet category",
            "published_psi_H020": "Published modelled site occupancy at H=0.20; probability 0-1, two decimal places",
            "published_psi_H040": "Published modelled site occupancy at H=0.40; probability 0-1, two decimal places",
            "published_H_psi060_percent": "Published forest-cover threshold for psi=0.60; percentage, one decimal place",
            "included_in_reconstruction": "True only for the 15 Forest-labelled rows",
        },
        "empirical_unit": "Sampled forest site; forest cover within a 600-m-radius buffer",
        "observed_H_range": [hmin, hmax], "range_source": "Supplement Table S1",
    }
    write_json(source_dir / "source_manifest.json", source_manifest)
    largest = max(coef_rows, key=lambda r: abs(r["H_psi060_difference_percentage_points"]))
    audit = {
        "status": "reconstructed_and_checked", "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version, "schema_version": cfg["schema_version"],
        "configuration": cfg,
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (config_path, extract, table, supplement, grid_config, Path(__file__).resolve())},
        "table1_rows": len(rows), "selected_species": len(selected), "excluded_non_forest_species": len(rows) - len(selected),
        "dense_H_count": len(dense_h), "dense_curve_rows": len(dense_rows),
        "main_H_total": len(all_main_h), "main_H_in_observed_domain": len(main_h), "main_curve_rows": len(main_rows),
        "main_H_excluded_outside_observed_domain": [h for h in all_main_h if h not in main_h],
        "max_anchor_probability_error": max_anchor_error,
        "max_independent_curve_probability_error": max_independent_error,
        "threshold_rounding_compatible_species": sum(r["threshold_compatible_with_rounding"] for r in coef_rows),
        "largest_threshold_difference": {k: largest[k] for k in ("species_as_published", "H_psi060_difference_percentage_points", "published_H_psi060", "reconstructed_H_psi060")},
        "empirical_ci_available": False,
        "checks": ["Counts and unique keys", "Published probability bounds", "Supplement Table S2 classifications and guilds", "Anchor reconstruction", "Independent odds-formula checks for every exported curve point", "Positive slopes and monotonic curves", "Observed-domain restriction", "Held-out threshold compatibility under rounding", "Saved CSV roundtrip"],
        "output_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p) for p in outputs.values()},
    }
    write_json(audit_dir / "11_bird_reconstruction_audit.json", audit)
    write_json(output_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "source_manifest": str((source_dir / "source_manifest.json").relative_to(ROOT)),
        "source_manifest_sha256": sha256(source_dir / "source_manifest.json"),
        "audit_path": str(audit_dir / "11_bird_reconstruction_audit.json"),
        "method": cfg["coefficient_method"], "output_sha256": audit["output_sha256"],
        "interpretation": "Approximate central empirical response curves. Rounding envelopes in the coefficient table are deterministic diagnostics, not empirical confidence intervals. The coefficients are not demographic r or dispersal parameters.",
    })
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, help="Optional separate verification output")
    args = parser.parse_args()
    result = run(args.config.resolve(), args.output_dir.resolve() if args.output_dir else None)
    print(json.dumps({k: result[k] for k in ("status", "table1_rows", "selected_species", "dense_curve_rows", "main_curve_rows", "max_anchor_probability_error", "max_independent_curve_probability_error", "threshold_rounding_compatible_species", "largest_threshold_difference")}, indent=2))
