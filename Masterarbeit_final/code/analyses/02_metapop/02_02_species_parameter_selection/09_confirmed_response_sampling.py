"""Area-stratified conditional sampling of the verified inner response domain."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(FINAL/'code/base'))
import numpy as np
import pandas as pd
from src.metapop.response_domain import quantile_bounds,envelope_certificate
DEFAULT=FINAL/'config/02_metapop/02_02_species_parameter_selection/confirmed_response_sampling/09_confirmed_response_sampling.json'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def inner_strips(ell,lr,categories,ell_window,lr_window):
    """Exact stepwise rectangles in log coordinates; no surface interpolation."""
    ell,lr,categories=np.asarray(ell),np.asarray(lr),np.asarray(categories)
    breaks=np.unique(np.r_[np.log(ell_window),np.log(ell[(ell>ell_window[0])&(ell<ell_window[1])])])
    strips=[]
    for a,b in zip(breaks[:-1],breaks[1:]):
        center=np.exp((a+b)/2)
        above=lr[(ell>=center)&np.isin(categories,[1,2])]
        below=lr[(ell<=center)&np.isin(categories,[0,1])]
        if not len(above) or not len(below):continue
        low=max(float(above.min()),float(lr_window[0]))
        high=min(float(below.max()),float(lr_window[1]))
        if low<high:strips.append(dict(log_ell_low=float(a),log_ell_high=float(b),log_r_low=low,
            log_r_high=high,log_area=float((b-a)*(high-low))))
    if not strips:raise ValueError('No certified inner domain')
    return pd.DataFrame(strips)


def area_map(strips,u_area,u_ratio):
    area=strips.log_area.to_numpy();cumulative=np.cumsum(area);total=cumulative[-1]
    if not 0<u_area<1 or not 0<u_ratio<1:raise ValueError('Interior unit coordinates required')
    target=u_area*total;i=int(np.searchsorted(cumulative,target,side='right'))
    row=strips.iloc[i];previous=cumulative[i-1] if i else 0.
    log_ell=row.log_ell_low+(target-previous)/(row.log_r_high-row.log_r_low)
    log_r=row.log_r_low+u_ratio*(row.log_r_high-row.log_r_low)
    return float(log_ell),float(log_r),i


def run(path):
    c=json.loads(path.read_text());bd=FINAL/c['boundary_data'];audit_path=FINAL/c['boundary_audit']
    audit=json.loads(audit_path.read_text());assert audit['status']=='independently_audited'
    bc=json.loads((FINAL/c['boundary_configuration']).read_text())
    receipt_path=FINAL/bc['results_directory']/'logs/09_compute_receipt.json'
    receipt=json.loads(receipt_path.read_text());assert sha(receipt_path)==audit['compute_receipt_sha256']
    for name,digest in receipt['outputs_sha256'].items():assert sha(FINAL/name)==digest
    array=bd/'09_combined_quantile_bounds.npz'
    with np.load(array) as z:e,lr,cat,lo,hi=z['half_distance'],z['log_r'],z['category'],z['lower'],z['upper']
    window=np.asarray(c['log10_r_window'])*np.log(10)
    strips=inner_strips(e,lr,cat,c['half_distance_window'],window)
    total=float(strips.log_area.sum())
    rng=np.random.default_rng(c['seed']);nx=c['n_dispersion_strata'];ny=c['n_ratio_strata_per_dispersion']
    records=[];min_margin=1.
    for ix in range(nx):
        u_area=(ix+rng.random())/nx
        for iy in range(ny):
            u_ratio=(iy+rng.random())/ny;le,rate,segment=area_map(strips,u_area,u_ratio)
            ell=np.exp(le);qlo,qhi=quantile_bounds(e,lr,lo,hi,ell,rate,bc['distance_upper_bound'])
            cert=envelope_certificate(qlo,qhi,c['saturation_threshold']);assert cert['category']==1
            min_margin=min(min_margin,.95-cert['L_upper'],cert['U_lower']-.05)
            records.append(dict(parameter_id=f'domain{len(records)+1:03d}',dispersion_group=ix,
                dispersion_stratum=ix,ratio_stratum=iy,unit_area=u_area,unit_conditional_ratio=u_ratio,
                log_r=rate,log10_r=rate/np.log(10),r=float(np.exp(rate)),half_distance=float(ell),
                alpha=float(np.log(2)/ell),q=0.,parameter_design_weight=1/(nx*ny),
                domain_segment=segment,retention_L_upper=cert['L_upper'],retention_U_lower=cert['U_lower']))
    table=pd.DataFrame(records);assert len(table)==256 and not table.duplicated(['half_distance','r']).any()
    assert table.groupby('dispersion_group').size().eq(ny).all()
    assert table.half_distance.between(*c['half_distance_window']).all() and table.log10_r.between(*c['log10_r_window']).all()
    np.testing.assert_array_equal(np.floor(table.groupby('dispersion_group').unit_area.first()*nx).astype(int),np.arange(nx))
    data,results=FINAL/c['data_directory'],FINAL/c['results_directory']
    for p in [data,results/'logs',results/'tables']:p.mkdir(parents=True,exist_ok=True)
    sources=[path,Path(__file__),audit_path,receipt_path,array,FINAL/'code/base/src/metapop/response_domain.py']
    hashes={str(p.relative_to(FINAL)):sha(p) for p in sources}
    fp=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    manifest=dict(configuration=c,source_sha256=hashes,design_fingerprint=fp)
    marker=data/'manifest.json'
    if marker.exists():assert json.loads(marker.read_text())==manifest,'Changed design inputs; overwrite refused'
    marker.write_text(json.dumps(manifest,indent=2)+'\n')
    table.to_csv(data/'09_confirmed_response_parameters.csv',index=False)
    strips.to_csv(data/'09_confirmed_log_area_strips.csv',index=False)
    rectangle_area=float(np.log(c['half_distance_window'][1]/c['half_distance_window'][0])*(window[1]-window[0]))
    preflight=dict(status='design_verified',n_parameters=len(table),n_dispersion_groups=nx,n_ratios_per_group=ny,
        same_kernel_ratio_groups=True,log_area=total,original_rectangle_log_area=rectangle_area,
        retained_area_fraction=total/rectangle_area,point_density_ratio_vs_original_128=2*rectangle_area/total,
        minimum_retention_certificate_margin=min_margin,design_fingerprint=fp,
        half_distance_min=float(table.half_distance.min()),half_distance_max=float(table.half_distance.max()),
        r_min=float(table.r.min()),r_max=float(table.r.max()),
        source_sha256=hashes,output_sha256={str(p.relative_to(FINAL)):sha(p) for p in data.glob('*.csv')})
    (results/'logs/09_design_preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
    (results/'09_sampling_methods.md').write_text(f'''# Confirmed-domain sampling protocol

The user authorised overnight sampling and equilibrium calculations on
13 October 2026. We chose 256 points in the monotonically certified inner
response region within the previously justified main rectangle: half-distance
1–512 cells and log10(r)=-2.5 to 5.75. The expanded 1–4096-cell boundary search
was kept as a diagnostic. Its low-r tail is driven by isolated patches at
H=.01 and was not automatically adopted as the main sampling distribution.
The dispersal plateau remains outside the main domain.

The target measure is uniform area in (log half-distance, log r), conditional
on both retention inequalities being certified. Exact stepwise strips describe
the inner region without interpolation. Their total logarithmic area is
{total:.6g}, or {total/rectangle_area:.3%} of the original rectangle. We sampled
one independently jittered marginal-area coordinate in each of 64 equal-area
strata, then four independently jittered log-r quartiles conditional on each
dispersal coordinate. The inverse-area mapping weights wide parts of the
region proportionally to their area rather than assigning equal weight to
arbitrary dispersal columns. Every parameter point has design weight 1/256.
The random seed is {c['seed']}. Conditional stratification shares a kernel
among four ratios while retaining one fixed parameter combination per
landscape trajectory. This is a conditional stratified design, not a
two-dimensional Latin-hypercube design. Its point density is
{2*rectangle_area/total:.3g} times the original 128-point rectangular design.

All 256 points were checked against the same L<.95/U>.05 quantile-envelope
criterion. Unresolved boundary gaps are excluded from this conservative inner
domain and remain documented in the boundary pilot. We did not select points
by bird RMSE or association strength. Earlier LHS points and adaptively
selected boundary queries remain separate; they are not pooled unweighted
into the new main parameter summaries. Thus inference is conditional on
this explicitly defined target region. This selection does not identify
biological species parameters or require a complete 1-to-0 habitat response.

The occupancy run starts with five matched landscape seeds across all seven
scenarios and all 88 habitat levels. Further complete matched seed blocks
are added only after runtime and seed-stability checks, up to 20. Parameter
resolution and landscape replication are distinct; five seeds remain a
preliminary dataset, not evidence of precise intervals in every cell.
''')
    print(json.dumps({k:preflight[k] for k in ['status','n_parameters','retained_area_fraction','point_density_ratio_vs_original_128','minimum_retention_certificate_margin','r_min','r_max']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=DEFAULT)
    args=parser.parse_args();run(args.config.resolve())
