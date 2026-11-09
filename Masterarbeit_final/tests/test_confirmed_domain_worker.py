import importlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np
import pandas as pd
FINAL=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(FINAL/'code/analyses/02_metapop/02_03_main_analyses/02_03_02_species_dependence'))
V=importlib.import_module('11_confirmed_domain_worker')
M=importlib.import_module('11_confirmed_domain_compute')


class ConditionalWorkerChecks(unittest.TestCase):
    def test_group_and_independent_reference_known_equilibria(self):
        c=json.loads((FINAL/'config/02_metapop/02_03_main_analyses/02_03_02_species_dependence/confirmed_domain_256/11_confirmed_domain_occupancy.json').read_text())
        items=pd.DataFrame(dict(parameter_id=['domain001','domain002','domain003','domain004'],half_distance=[1.]*4,
            alpha=[np.log(2)]*4,log_r=np.log([.1,1,2,8])))
        values,reports=V.group(np.array([4.,4.]),np.array([[0.,1.],[1.,0.]]),items,c)
        np.testing.assert_allclose(values,[[.975,.75,.5,0]]*2,atol=1e-10)
        self.assertTrue(all(r['independent_check_converged'] for r in reports))
        self.assertTrue(all(r['fixed_point_residual']<=1.1e-12 for r in reports))

    def test_one_patch_cache_keeps_zero_length_vectors_and_missing_occupancy(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'fixture.csv';ids=[f'domain{i:03d}' for i in range(1,257)]
            pd.DataFrame(dict(parameter_id=ids,n_patches=1,occupancy=np.nan,status='excluded_fewer_than_two_patches')).to_csv(file,index=False)
            np.savez_compressed(file.with_suffix('.npz'),probabilities=np.array([]),offsets=np.zeros(257,dtype=int))
            V.R.save_json(file.with_suffix('.json'),dict(producer_fingerprint='fixture',grid_sha256='grid',
                hashes={s:V.R.sha(file.with_suffix(s)) for s in ['.csv','.npz']}))
            table=V.snapshot_cache(file,'fixture','grid',ids)
            self.assertTrue(table.occupancy.isna().all())
            np.savez_compressed(file.with_suffix('.npz'),probabilities=np.array([]),offsets=np.arange(257,dtype=int))
            V.R.save_json(file.with_suffix('.json'),dict(producer_fingerprint='fixture',grid_sha256='grid',
                hashes={s:V.R.sha(file.with_suffix(s)) for s in ['.csv','.npz']}))
            with self.assertRaises(AssertionError):V.snapshot_cache(file,'fixture','grid',ids)

    def test_expired_optional_deadline_does_not_start_model(self):
        with self.assertRaises(TimeoutError):M.deadline_worker(time.time()-1)


if __name__=='__main__':unittest.main()
