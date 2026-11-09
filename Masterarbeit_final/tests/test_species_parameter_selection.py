"""Numerical checks against known equilibria and the established solver."""
import unittest
import importlib.util
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from src.metapop.parameter_sweep import continuous_ratio_sweep
from src.metapop.hanski import run_continuous_equilibrium_from_kernel
from src.landscape.landscape import Landscape


class RatioSweepTests(unittest.TestCase):
    def test_two_equal_patches_have_known_positive_and_zero_equilibria(self):
        a = np.array([4., 4.])
        k = np.array([[0., .5], [.5, 0.]])
        results = continuous_ratio_sweep(a, k, [1., 8.])
        np.testing.assert_allclose(results[0].probabilities, .75, atol=1e-9)
        np.testing.assert_allclose(results[1].probabilities, 0, atol=1e-9)
        self.assertTrue(all(v.converged for v in results))

    def test_matches_scalar_solver_and_rate_balance_on_unequal_patches(self):
        a = np.array([1., 3., 9., 2.])
        k = np.array([[0,.4,.2,.8],[.4,0,.7,.1],[.2,.7,0,.3],[.8,.1,.3,0]])
        ratios = [.1, 1., 10., 100.]
        for r, actual in zip(ratios, continuous_ratio_sweep(a, k, ratios)):
            expected = run_continuous_equilibrium_from_kernel(a, k, r=r, b=1, x=.5,
                                                             max_iter=20000, tol=1e-10)
            np.testing.assert_allclose(actual.probabilities, expected.probabilities, atol=2e-10)
            self.assertEqual(actual.converged, expected.converged)
            p = actual.probabilities
            balance = (1-p)*(k@(a*p)) - r*a**(-.5)*p
            np.testing.assert_allclose(balance, 0, atol=1e-8)

    def test_nonconvergence_is_retained(self):
        value = continuous_ratio_sweep(np.ones(2), np.array([[0.,1.],[1.,0.]]),
                                       [1.], max_iter=1)[0]
        self.assertFalse(value.converged)
        self.assertEqual(value.n_iter, 1)

    def test_diagonal_patch_distance_is_sqrt_two_under_production_geometry(self):
        landscape = Landscape(np.eye(2, dtype=np.int8))
        landscape.ensure_clusters(edge_depth=0, connectivity='4', boundary_mode='open',
                                  distance_mode='voronoi_dt', voronoi_depth=1)
        self.assertEqual(landscape.n_clusters, 2)
        self.assertAlmostEqual(landscape.cluster_dist_matrix[0,1], np.sqrt(2))


def load_analysis(name):
    root = Path(__file__).resolve().parents[1]
    final = root if root.name == 'Masterarbeit_final' else root / 'Masterarbeit_final'
    path = final / 'code/analyses/02_metapop/02_02_species_parameter_selection' / name
    spec = importlib.util.spec_from_file_location(name.removesuffix('.py'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ParameterDesignTests(unittest.TestCase):
    def test_extension_merge_preserves_probability_vectors_and_provenance(self):
        module = load_analysis('09_expand_species_parameters.py')
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            parts=[]
            cases=[[(1.,1.,[.3,.7])],[(2.,1.,[.2,.4])],[(1.,2.,[.5,.9]),(2.,2.,[.4,.8])]]
            expected=[]
            for number,values in enumerate(cases):
                stem=directory/f'part{number}'
                records=[]; offsets=[0]; vectors=[]
                for r,ell,p in values:
                    records.append(dict(H_target=.5,r=r,half_distance=ell,n_patches=2,
                        status='valid',occupancy=np.mean(p),grid_sha256='same_grid',producer_fingerprint=f'original{number}'))
                    vectors.extend(p); offsets.append(len(vectors)); expected.extend(p)
                pd.DataFrame(records).to_csv(stem.with_suffix('.csv'),index=False)
                np.savez_compressed(stem.with_suffix('.npz'),probabilities=vectors,offsets=offsets)
                parts.append(stem)
            c=dict(r_values=[1.,2.],half_distance_values=[1.,2.],h_targets=[.5],area_units='cells')
            destination=directory/'merged'
            merged=module.combine(parts,destination,c,'expanded')
            self.assertEqual(len(merged),4)
            self.assertTrue(merged.producer_fingerprint.eq('expanded').all())
            self.assertEqual(merged.source_producer_fingerprint.tolist(),['original0','original1','original2','original2'])
            with np.load(destination.with_suffix('.npz')) as saved:
                np.testing.assert_array_equal(saved['probabilities'],expected)
                np.testing.assert_array_equal(saved['offsets'],[0,2,4,6,8])

    def test_profiles_follow_quartiles_in_log_space(self):
        module = load_analysis('09_species_parameter_selection.py')
        profiles = module.profile_table({'r':[1.,10000.], 'half_distance':[1.,16.]})
        reference = profiles.iloc[0]
        self.assertAlmostEqual(reference.r, 100.)
        self.assertAlmostEqual(reference.half_distance, 4.)
        self.assertAlmostEqual(reference.alpha, np.log(2)/4)
        self.assertAlmostEqual(profiles.iloc[1].r, 10.)
        self.assertAlmostEqual(profiles.iloc[1].half_distance, 2.)
        self.assertAlmostEqual(profiles.iloc[4].r, 1000.)
        self.assertAlmostEqual(profiles.iloc[4].half_distance, 8.)

    def test_exclusions_and_nonconvergence_do_not_count_as_zero(self):
        module = load_analysis('09_species_parameter_selection.py')
        data = pd.DataFrame([
            dict(scenario_id='s',H_target=.5,r=1.,half_distance=2.,n_patches=2,status='valid',occupancy=.5),
            dict(scenario_id='s',H_target=.5,r=1.,half_distance=2.,n_patches=1,status='excluded_fewer_than_two_patches',occupancy=np.nan),
            dict(scenario_id='s',H_target=.5,r=1.,half_distance=2.,n_patches=2,status='nonconverged',occupancy=np.nan)])
        row = module.diagnostics(data,{'selection':{'floor':.05,'ceiling':.95}}).iloc[0]
        self.assertEqual(row.n_valid,1)
        self.assertEqual(row.n_excluded,1)
        self.assertEqual(row.n_nonconverged,1)
        self.assertEqual(row['median'],.5)
        self.assertEqual(row.floor_fraction,0)

    def test_transition_interpolation_cannot_bridge_an_undefined_level(self):
        module = load_analysis('09_species_parameter_summary.py')
        point, count = module.first_downward_crossing(np.array([.2,.5,.8]),
                                                     np.array([.1,np.nan,.9]),.5)
        self.assertTrue(np.isnan(point))
        self.assertEqual(count,0)
        point,count=module.first_downward_crossing(np.array([.2,.5,.8]),np.array([.1,.5,.9]),.5)
        self.assertAlmostEqual(point,.5)
        self.assertEqual(count,1)


if __name__ == '__main__':
    unittest.main()
