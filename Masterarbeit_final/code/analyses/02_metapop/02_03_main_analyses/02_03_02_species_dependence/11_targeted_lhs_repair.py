"""Diagnose a failed initial-LHS fixture, then resume only its verified trajectory.

The primary solver/configuration remain unchanged. A repair is permitted only
after reproducing a rejected independent comparison and verifying it with a
tighter independent reference, without replacing any landscape seed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
FINAL = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
FOLDER = Path(__file__).parent
sys.path.insert(0, str(FOLDER)); sys.path.insert(0, str(FINAL/'code/base'))
import numpy as np
import pandas as pd
W = importlib.import_module('11_lhs_workers')
REL = Path('02_metapop/02_03_main_analyses/02_03_02_species_dependence/lhs_20_seeds')
DATA = FINAL/'data/processed'/REL
RESULTS = FINAL/'results'/REL
OUT = RESULTS/'logs/targeted_repairs'


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def install_deadline():
    deadline = datetime(2026, 10, 13, 5, tzinfo=timezone.utc).timestamp()
    remaining = deadline-time.time()
    if remaining <= 0:
        raise TimeoutError('The authorised compute cutoff has passed.')
    def stop(_signal, _frame):
        raise TimeoutError('Targeted repair reached the compute cutoff.')
    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, remaining)


def inputs(sid, rep):
    m = json.loads((DATA/'manifest.json').read_text())
    for name, digest in m['source_hashes'].items():
        assert hashlib.sha256((FINAL/name).read_bytes()).hexdigest() == digest, name
    c = m['configuration']
    assert rep in c['replicates']
    lc = json.loads((FINAL/c['landscape_config']).read_text())
    scenario = next(s for s in lc['scenarios'] if s['scenario_id'] == sid)
    benchmark = pd.read_csv(FINAL/c['landscape_data'])
    benchmark = benchmark[benchmark.scenario_id.eq(sid) & benchmark.replicate.eq(rep)]
    lhs = pd.read_csv(FINAL/c['lhs_design'])
    assert len(lhs) == 128 and len(benchmark) == 88
    return m, c, lc, scenario, benchmark, lhs


def diagnose(sid, rep):
    m, c, lc, scenario, benchmark, lhs = inputs(sid, rep)
    identity = f'{sid}__rep{rep:03d}'
    assert not (DATA/'trajectories'/f'{identity}.csv').exists(), 'Already completed; no repair needed.'
    progress = json.loads((DATA/'progress'/f'{identity}.json').read_text())
    completed = progress['completed_snapshots']
    h = sorted(c['h_targets'], reverse=True)[completed]
    assert not (DATA/'snapshots'/f'{identity}__h{h:.3f}.csv').exists()
    seed = lc['base_seed']+rep-1
    pars = lc['access_reference']
    run = W.run_sequential_deforestation(np.ones((200, 200), dtype=np.int8), np.random.default_rng(seed),
        rho=scenario['rho'], directed_mode=scenario['mode'], target_habitat_fraction=0.,
        generation_boundary=lc['generation_boundary'], eta=pars['eta'], k=pars['k'],
        lambda_decay=pars['lambda'], save_habitat_fractions=c['h_targets'], seed=seed)
    snap = next(s for s in run.snapshots if float(s.requested_habitat_fractions[0]) == h)
    grid_hash = hashlib.sha256(snap.grid.tobytes()).hexdigest()
    old = benchmark[benchmark.H_target.eq(h)].iloc[0]
    assert grid_hash == old.grid_sha256 and snap.event_count == old.event_count
    land = W.Landscape(snap.grid)
    land.ensure_clusters(edge_depth=0, connectivity='4', boundary_mode='open',
                         distance_mode='voronoi_dt', voronoi_depth=1)
    a = np.asarray(land.cluster_areas, float)
    d = np.asarray(land.cluster_dist_matrix, float)
    assert len(a) == old.n_clusters and a.sum() == round(40000*h)
    assert np.isfinite(d).all() and (d >= 0).all() and np.array_equal(d, d.T)
    original = W.scaled_ratio_sweep
    latest = {}
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        latest['reference'] = result[0]
        return result
    W.scaled_ratio_sweep = capture
    failure = None
    try:
        for item in lhs.itertuples(index=False):
            latest.clear()
            try:
                W.solve(a, d, item, c)
            except AssertionError as error:
                reference = latest.get('reference')
                if reference is None or not reference.converged:
                    raise RuntimeError('Failure is not a verified independent-reference comparison.') from error
                primary, report = W.solve(a, d, item, {**c, 'scalar_check_ids': []})
                difference = float(np.max(np.abs(primary-reference.probabilities)))
                assert report['status'] == 'valid'
                assert difference > c['solver']['independent_probability_tolerance']
                def tight(*args, **kwargs):
                    kwargs['tol'] = min(kwargs['tol'], 1e-12)
                    result = original(*args, **kwargs)
                    latest['tight_reference'] = result[0]
                    return result
                W.scaled_ratio_sweep = tight
                checked, tight_report = W.solve(a, d, item, c)
                assert latest['tight_reference'].converged
                np.testing.assert_array_equal(primary, checked)
                assert tight_report['independent_probability_difference'] <= c['solver']['independent_probability_tolerance']
                fixture = OUT/f'{identity}__fixture.npz'
                np.savez_compressed(fixture, areas=a, distances=d, probabilities=primary,
                    original_reference=reference.probabilities,
                    tighter_reference=latest['tight_reference'].probabilities)
                failure = dict(status='independent_reference_precision_rejection_verified',
                    trajectory=identity, scenario_id=sid, replicate=rep, seed=seed, H=h,
                    parameter_id=item.parameter_id, r=float(np.exp(item.log_r)),
                    half_distance=float(item.half_distance), n_patches=len(a), grid_sha256=grid_hash,
                    original_difference=difference,
                    tighter_difference=tight_report['independent_probability_difference'],
                    primary_occupancy=report['occupancy'], primary_solver=report['solver'],
                    primary_residual=report['fixed_point_residual'], primary_solver_and_tolerance_unchanged=True,
                    original_primary_producer_fingerprint=m['producer_fingerprint'],
                    supplementary_verification_source_sha256=source_hash(),
                    fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                    as_of_utc=datetime.now(timezone.utc).isoformat())
                break
    finally:
        W.scaled_ratio_sweep = original
    if failure is None:
        raise RuntimeError('No comparison rejection reproduced; do not assume a precision repair.')
    W.atomic_json(OUT/f'{identity}__diagnosis.json', failure)
    print(json.dumps(failure, indent=2), flush=True)


def resume(sid, rep):
    m, c, lc, scenario, benchmark, lhs = inputs(sid, rep)
    identity = f'{sid}__rep{rep:03d}'
    assert not (DATA/'trajectories'/f'{identity}.csv').exists(), 'Already completed; preserving valid data.'
    diagnosis = json.loads((OUT/f'{identity}__diagnosis.json').read_text())
    assert diagnosis['status'] == 'independent_reference_precision_rejection_verified'
    assert diagnosis['supplementary_verification_source_sha256'] == source_hash()
    assert diagnosis['original_primary_producer_fingerprint'] == m['producer_fingerprint']
    assert hashlib.sha256((OUT/f'{identity}__fixture.npz').read_bytes()).hexdigest() == diagnosis['fixture_sha256']
    original = W.scaled_ratio_sweep
    def accurate(*args, **kwargs):
        kwargs['tol'] = min(kwargs['tol'], 1e-12)
        return original(*args, **kwargs)
    W.scaled_ratio_sweep = accurate
    original_solve = W.solve
    def recorded(*args, **kwargs):
        p, report = original_solve(*args, **kwargs)
        report['independent_reference_tolerance'] = 1e-12
        report['supplementary_verification_source_sha256'] = source_hash()
        return p, report
    W.solve = recorded
    # Reuse every verified snapshot; only the uncomputed suffix is produced.
    result = W.trajectory(scenario, rep, c, lc, benchmark, lhs, str(DATA), m['producer_fingerprint'])
    assert result[2] == 0
    note = dict(status='trajectory_resumed_and_verified', trajectory=identity,
        original_primary_producer_fingerprint=m['producer_fingerprint'],
        supplementary_verification_source_sha256=source_hash(),
        primary_solver_and_tolerance_unchanged=True, independent_reference_tolerance=1e-12,
        independent_probability_comparison_tolerance_unchanged=c['solver']['independent_probability_tolerance'],
        diagnosis_path=str((OUT/f'{identity}__diagnosis.json').relative_to(FINAL)),
        diagnosis_sha256=hashlib.sha256((OUT/f'{identity}__diagnosis.json').read_bytes()).hexdigest(),
        result=list(result), as_of_utc=datetime.now(timezone.utc).isoformat())
    W.atomic_json(OUT/f'{identity}__verification.json', note)
    print(json.dumps(note, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['diagnose', 'resume'], required=True)
    p.add_argument('--scenario-id', required=True)
    p.add_argument('--replicate', type=int, required=True)
    args = p.parse_args()
    install_deadline()
    OUT.mkdir(parents=True, exist_ok=True)
    (diagnose if args.stage == 'diagnose' else resume)(args.scenario_id, args.replicate)


if __name__ == '__main__':
    main()
