"""Reject mismatched geometry even when a progress receipt is unavailable."""
import hashlib
import importlib
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT if ROOT.name=='Masterarbeit_final' else ROOT/'Masterarbeit_final'
sys.path.insert(0,str(FINAL/'code/analyses/02_metapop/02_03_main_analyses/02_03_02_species_dependence'))
P=importlib.import_module('11_partial_plausibility')


class ReplayedGeometryChecks(unittest.TestCase):
    def setUp(self):
        self.grid=np.array([[1,0,1]],dtype=np.int8)
        self.digest=hashlib.sha256(self.grid.tobytes()).hexdigest()
        self.areas=np.array([1.,1.])
        self.distances=np.array([[0.,2.],[2.,0.]])

    def test_known_two_patch_fixture_without_progress_receipt(self):
        a,d=P.verify_replayed_pilot_geometry(self.grid,self.digest,self.areas,self.distances)
        np.testing.assert_array_equal(a,self.areas)
        np.testing.assert_array_equal(d,self.distances)

    def test_wrong_grid_identity_is_rejected(self):
        with self.assertRaises(AssertionError):
            P.verify_replayed_pilot_geometry(self.grid,'wrong',self.areas,self.distances)

    def test_wrong_patch_areas_are_rejected(self):
        with self.assertRaises(AssertionError):
            P.verify_replayed_pilot_geometry(self.grid,self.digest,[2.,1.],self.distances)

    def test_wrong_distances_are_rejected(self):
        with self.assertRaises(AssertionError):
            P.verify_replayed_pilot_geometry(self.grid,self.digest,self.areas,self.distances/2)


if __name__=='__main__':unittest.main()
