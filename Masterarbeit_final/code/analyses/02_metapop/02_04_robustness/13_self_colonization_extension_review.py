"""All-H boundary validation, independent saved-vector audit and staged report."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('self_extension',HERE/'13_self_colonization_extension.py')
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
P,R=E.P,E.R
import numpy as np
import pandas as pd
from src.tools.final_plot_style import apply_final_style,figure_size,export_figure

def endpoint_task(row,c,design,data_name,fp,source_folders):
    data=Path(data_name);name=f'{row.scenario_id}__rep{int(row.replicate):03d}__h{row.H_target:.3f}'
    path=data/'validation'/f'{name}.csv';marker=path.with_suffix('.json')
    if marker.exists():
        meta=json.loads(marker.read_text());assert meta['fingerprint']==fp
        assert P.sha(path)==meta['csv_sha256'] and P.sha(path.with_suffix('.npz'))==meta['vectors_sha256'];return
    a,d=E.geometry(row,c);values=np.empty((len(a),len(design)));reports={};retained_sources={}
    for source_folder in source_folders:
        source=Path(source_folder)/path.name
        if not source.exists():continue
        sm=json.loads(source.with_suffix('.json').read_text())
        assert P.sha(source)==sm['csv_sha256'] and P.sha(source.with_suffix('.npz'))==sm['vectors_sha256']
        ss=pd.read_csv(source,float_precision='round_trip');assert ss.grid_sha256.eq(row.grid_sha256).all()
        with np.load(source.with_suffix('.npz')) as z:vv=z['probabilities']
        for i,item in design.iterrows():
            match=ss.index[ss.half_distance.eq(item.half_distance)&np.isclose(ss.r,item.r,rtol=1e-14,atol=0)]
            if len(match) and i not in reports:
                j=int(match[0]);values[:,i]=vv[:,j];report=ss.loc[j]
                reports[i]=dict(converged=True,n_iter=int(report.n_iter),fixed_point_residual=float(report.fixed_point_residual),solver='retained_verified_fixed_parameter_vector',independent_difference=None)
                retained_sources[str(source)]=P.sha(source)
    missing=design.loc[[i for i in design.index if i not in reports]]
    for ell,g in missing.groupby('half_distance'):
        p,report=R.recovered_group(a,d,g,c);values[:,g.index]=p
        reports.update(zip(g.index,report))
    records=[]
    for i,item in design.iterrows():
        records.append(dict(scenario_id=row.scenario_id,replicate=row.replicate,H_target=row.H_target,
            half_distance=item.half_distance,r=item.r,role=item.role,occupancy=float(values[:,i].mean()),
            grid_sha256=row.grid_sha256,n_patches=len(a),**reports[i]))
    pd.DataFrame(records).to_csv(path,index=False);np.savez_compressed(path.with_suffix('.npz'),probabilities=values)
    P.save(marker,dict(fingerprint=fp,csv_sha256=P.sha(path),vectors_sha256=P.sha(path.with_suffix('.npz')),retained_sources=retained_sources))

def endpoint_validation(c,domain,land,data,out,n,workers):
    stage=out/f'seeds{n:03d}';original=json.loads((stage/'logs/13_q1_domain.json').read_text())
    ell=[v for v in c['domain_half_distances'] if v<=original['half_distance_max']]
    lo,hi=original['log10_r_min'],original['log10_r_max'];history=[]
    selected=land[land.replicate.le(n)&land.n_clusters.ge(2)]
    sources={str(p.relative_to(P.FINAL)):P.sha(p) for p in [Path(__file__),Path(E.__file__),P.FINAL/'code/base/src/metapop/self_equilibrium.py']}
    for attempt in range(1,9):
        design=pd.DataFrame([dict(half_distance=e,r=10**lr,role=role) for e in ell for lr,role in [(lo,'lower'),(hi,'upper')]])
        fp=hashlib.sha256(json.dumps(dict(sources=sources,log10_r=[lo,hi],n=n),sort_keys=True).encode()).hexdigest()
        folder=data/'endpoint_validation'/f'seeds{n:03d}'/f'attempt{attempt:02d}'
        (folder/'validation').mkdir(parents=True,exist_ok=True)
        source_folders=[str(data/'domain')]
        if attempt>1:source_folders.append(str(folder.parent/f'attempt{attempt-1:02d}'/'validation'))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fs=[pool.submit(endpoint_task,SimpleNamespace(**row),c,design,str(folder),fp,source_folders) for row in selected.to_dict(orient='records')]
            for i,f in enumerate(as_completed(fs),1):
                f.result()
                if i%100==0:print(f'Endpoint validation {attempt}: {i}/{len(fs)}',flush=True)
        table=pd.concat([pd.read_csv(p) for p in (folder/'validation').glob('*.csv')],ignore_index=True)
        med=table.groupby(['scenario_id','H_target','half_distance','role']).occupancy.agg(n='count',median='median').reset_index()
        med=med[med.n>=c['minimum_valid_seeds']]
        coverage=med.assign(covered=np.where(med.role.eq('lower'),med['median']>=.95-1e-10,med['median']<=.05+1e-10)).groupby(['half_distance','role']).covered.agg(['mean','count']).reset_index()
        coverage.to_csv(stage/'tables'/f'13_endpoint_coverage_attempt{attempt:02d}.csv',index=False)
        low_pass=bool(coverage[coverage.role.eq('lower')]['mean'].ge(.95-1e-12).all())
        high_pass=bool(coverage[coverage.role.eq('upper')]['mean'].ge(.95-1e-12).all())
        history.append(dict(attempt=attempt,log10_r_min=lo,log10_r_max=hi,lower_pass=low_pass,upper_pass=high_pass,fingerprint=fp))
        if low_pass and high_pass:
            result=dict(original,log10_r_min=lo,log10_r_max=hi,r_min=10**lo,r_max=10**hi,
                status='working_domain_with_all_H_endpoint_coverage',n_endpoint_H_levels=int(med.H_target.nunique()),
                endpoint_rule='At every retained half-distance, >=95% of eligible scenario-H medians reach >=.95 at lower-r and <=.05 at upper-r; >=3 valid seeds per cell.',
                selection_scope='Plateau and initial central90% brackets use five H; endpoints validated on every eligible stored H. Conditional numerical working domain, not biological limits.',
                endpoint_history=history,validation_directory=str(folder.relative_to(P.FINAL)),supplemental_sources=sources)
            P.save(stage/'logs/13_q1_validated_domain.json',result)
            table.to_csv(stage/'tables/13_endpoint_responses.csv',index=False)
            return result
        if not low_pass:lo=max(-2.,lo-.25)
        if not high_pass:hi=min(7.,hi+.25)
    raise RuntimeError('Endpoint coverage not established within the declared refinement budget')

def independent_domain_audit(c,domain,land,data,out,n):
    stage=out/f'seeds{n:03d}';maximum=0.;mean_error=0.;count=0;independent_statistics=[]
    for folder,kind in [(data/'domain','domain'),(P.FINAL/json.loads((stage/'logs/13_q1_validated_domain.json').read_text())['validation_directory']/'validation','endpoint')]:
        files=[p for p in folder.glob('*.csv') if int(p.name.split('__rep')[1][:3])<=n]
        if kind=='domain':assert len(files)==35*n
        for path in sorted(files):
            meta=json.loads(path.with_suffix('.json').read_text());assert P.sha(path)==meta['csv_sha256'] and P.sha(path.with_suffix('.npz'))==meta['vectors_sha256']
            table=pd.read_csv(path,float_precision='round_trip');row=table.iloc[0]
            if kind=='domain':
                assert table.parameter_id.tolist()==domain.parameter_id.tolist()
                np.testing.assert_allclose(table.r,domain.r,rtol=1e-14,atol=0)
                np.testing.assert_array_equal(table.half_distance,domain.half_distance)
            geometry_row=land[(land.scenario_id==row.scenario_id)&(land.replicate==row.replicate)&land.H_target.eq(row.H_target)].iloc[0]
            a,d=E.geometry(geometry_row,c)
            with np.load(path.with_suffix('.npz')) as z:values=z['probabilities']
            assert values.shape==(len(a),len(table)) and np.isfinite(values).all() and (values>=0).all() and (values<=1).all()
            err=float(np.max(np.abs(values.mean(axis=0)-table.occupancy)));mean_error=max(mean_error,err);assert err<1e-14
            for ell,g in table.groupby('half_distance'):
                p=values[:,g.index];k=np.exp(-np.log(2)/ell*d);np.fill_diagonal(k,1)
                source=k@(a[:,None]*p);target=source/(source+a[:,None]**(-.5)*g.r.to_numpy())
                residual=float(np.max(np.abs(target-p)));maximum=max(maximum,residual);assert residual<=1.1*c['tol']
            count+=len(table)
    # Literal ordered seed medians and scalar interpolation reproduce selected crossings.
    tt=pd.read_csv(stage/'tables/13_q1_transitions.csv')
    for scenario,h,ell in [('only_random_rho0',.2,1),('access_weighted_rho0999',.4,16),('edge_random_rho09',.6,256)]:
        curves=[]
        for rep in range(1,n+1):
            t=pd.read_csv(data/'domain'/f'{scenario}__rep{rep:03d}__h{h:.3f}.csv')
            curves.append(t[t.half_distance.eq(ell)].sort_values('log10_r').occupancy.to_numpy())
        matrix=np.sort(np.array(curves),axis=0)
        med=matrix[n//2] if n%2 else (matrix[n//2-1]+matrix[n//2])/2
        lr=np.array(c['domain_log10_ratios']);i=np.flatnonzero((med[:-1]>=.5)&(med[1:]<=.5))[0]
        crossing=float(lr[i]+(.5-med[i])*(lr[i+1]-lr[i])/(med[i+1]-med[i]))
        recorded=float(tt[(tt.scenario_id==scenario)&tt.H_target.eq(h)&tt.half_distance.eq(ell)].p50.iloc[0])
        assert abs(crossing-recorded)<1e-13;independent_statistics.append(abs(crossing-recorded))
    fixtures=[]
    for scenario,h in [('only_random_rho0',.2),('only_random_rho0',.4),('access_weighted_rho0999',.4)]:
        row=land[(land.scenario_id==scenario)&(land.replicate==n)&land.H_target.eq(h)].iloc[0]
        a,d=E.geometry(row,c);cache=P.FINAL/c['geometry_cache']/'landscapes'/f'{scenario}__rep{n:03d}.npz'
        with np.load(cache) as z:grid=z['grids'][np.flatnonzero(z['H']==h)[0]]
        assert hashlib.sha256(grid.tobytes()).hexdigest()==row.grid_sha256
        landscape=E.V.Landscape(grid);landscape.ensure_clusters(edge_depth=0,connectivity='4',boundary_mode='open',distance_mode='voronoi_dt',voronoi_depth=1)
        np.testing.assert_array_equal(landscape.cluster_areas,a);np.testing.assert_allclose(landscape.cluster_dist_matrix,d,rtol=0,atol=1e-12)
        fixtures.append(dict(scenario_id=scenario,replicate=n,H_target=h))
    P.save(stage/'logs/13_domain_independent_audit.json',dict(status='passed',n_vectors=count,maximum_original_residual=maximum,
        maximum_stored_mean_error=mean_error,maximum_independent_crossing_error=max(independent_statistics),source_sha256=P.sha(Path(__file__))))
    P.save(stage/'logs/13_new_seed_geometry_audit.json',dict(status='passed',fixtures=fixtures))

def omission_summary(c,d,land,data,out,n):
    stage=out/f'seeds{n:03d}';raw=pd.concat([pd.read_csv(p) for p in E.view(data,n).joinpath('snapshots').glob('*.csv')],ignore_index=True)
    raw=raw[raw['sample'].eq('matched')]
    def medians(part):
        a=part.groupby(['parameter_id','scenario_id','H_target']).agg(median=('occupancy','median'),count=('occupancy','count'))
        a.loc[a['count']<3,'median']=np.nan
        base=a.xs('only_random_rho0',level='scenario_id')['median']
        a['contrast']=a['median']-[base.loc[(i[0],i[2])] for i in a.index]
        return a
    full=medians(raw);rows=[]
    for seed in range(1,n+1):
        omitted=medians(raw[raw.replicate!=seed]);values={}
        for col in ['median','contrast']:
            delta=(omitted[col]-full[col]).abs().dropna();values[col+'_maximum']=float(delta.max());values[col+'_fraction_over_005']=float((delta>.05).mean())
        rows.append(dict(omitted_seed=seed,**values))
    pd.DataFrame(rows).to_csv(stage/'tables/13_matched_seed_omission.csv',index=False)
    # Omit an entire shared seed in all scenarios/H/profiles, retaining equal
    # scenario mass. Parameter profiles are never resampled or dropped.
    rank_rows=[];sensitivity_rows=[]
    for q,column in [(0,'baseline_occupancy'),(1,'occupancy')]:
        changes={seed:[] for seed in range(1,n+1)};lost={seed:0 for seed in range(1,n+1)}
        for h,g in raw.groupby('H_target'):
            wide=g.pivot(index='parameter_id',columns=['scenario_id','replicate'],values=column).reindex(index=d.parameter_id)
            columns=wide.columns;groups=columns.get_level_values('scenario_id').to_numpy();seeds=columns.get_level_values('replicate').to_numpy();y=wide.to_numpy()
            metrics=land[(land.H_target==h)&(land.replicate<=n)].set_index(['scenario_id','replicate']).reindex(columns)
            for metric in P.METRICS:
                x=metrics[metric].to_numpy();full=E.balanced_spearman(x,y,groups)
                rounded=E.balanced_spearman(x,np.round(y,8),groups)
                valid=np.isfinite(y);span=np.max(np.where(valid,y,-np.inf),axis=1)-np.min(np.where(valid,y,np.inf),axis=1)
                for i,pid in enumerate(d.parameter_id):
                    sensitivity_rows.append(dict(q=q,parameter_id=pid,H_target=h,metric=metric,
                        coefficient=full[i],rounded_coefficient=rounded[i],response_span=span[i] if np.isfinite(span[i]) else np.nan))
                for seed in range(1,n+1):
                    other=E.balanced_spearman(x,y,groups,frequencies=(seeds!=seed).astype(float))
                    delta=np.abs(other-full);changes[seed].extend(delta[np.isfinite(delta)].tolist())
                    lost[seed]+=int((np.isfinite(full)&~np.isfinite(other)).sum())
        for seed,values in changes.items():
            v=np.array(values);rank_rows.append(dict(q=q,omitted_seed=seed,n_comparable=len(v),n_newly_undefined=lost[seed],maximum_change=float(v.max()),fraction_over_005=float((v>.05).mean())))
    pd.DataFrame(rank_rows).to_csv(stage/'tables/13_D_seed_omission.csv',index=False)
    ss=pd.DataFrame(sensitivity_rows);ss.to_csv(stage/'tables/13_D_numerical_response_sensitivity.csv',index=False)
    primary=pd.read_csv(stage/'tables/13_D_metric_associations.csv')
    paired_primary=ss.merge(primary,on=['q','parameter_id','H_target','metric'],suffixes=('_assembled','_primary'),validate='one_to_one')
    assert len(paired_primary)==len(ss)==len(primary)
    np.testing.assert_allclose(paired_primary.coefficient_assembled,paired_primary.coefficient_primary,atol=1e-14,rtol=0,equal_nan=True)
    paired=ss.pivot(index=['parameter_id','H_target','metric'],columns='q',values=['coefficient','response_span'])
    good=paired.dropna(subset=[('coefficient',0),('coefficient',1)])
    informative=good[(good['response_span',0]>.05)&(good['response_span',1]>.05)]
    delta=(ss.coefficient-ss.rounded_coefficient).abs().dropna()
    P.save(stage/'logs/13_D_sensitivity.json',dict(minimum_response_range_both_models=.05,
        n_informative_paired=len(informative),median_absolute_model_change=float((informative['coefficient',1]-informative['coefficient',0]).abs().median()),
        opposite_sign_above_01=int(((informative['coefficient',0]*informative['coefficient',1]<0)&(informative['coefficient',0].abs()>.1)&(informative['coefficient',1].abs()>.1)).sum()),
        rounding_decimals=8,n_finite_rounding_pairs=len(delta),rounding_maximum_change=float(delta.max()),
        rounding_fraction_over_005=float((delta>.05).mean()),rounding_newly_undefined=int((ss.coefficient.notna()&ss.rounded_coefficient.isna()).sum())))

def report(c,d,land,data,out,n):
    stage=out/f'seeds{n:03d}';decision=json.loads((stage/'logs/13_q1_validated_domain.json').read_text());audit=json.loads((stage/'logs/13_numeric_audit.json').read_text())
    sensitivity=pd.read_csv(stage/'tables/13_domain_sensitivity.csv');omission=sensitivity[sensitivity.check.eq('complete_seed_omission')]
    summary=pd.read_csv(stage/'tables/13_B_reference.csv');p=summary.pivot(index=['scenario_id','H_target'],columns='q',values='contrast');difference=(p[1]-p[0]).abs()
    ordinary=float(difference[difference.index.get_level_values('H_target')<=.825].max());high=float(difference.max())
    ds=json.loads((stage/'logs/13_D_sensitivity.json').read_text())
    text=f'''# Expanded self-colonization comparison and working parameter domain

The comparison now uses {n} complete shared seed blocks, all seven scenarios,
13 stored habitat proportions and the unchanged 65 parameter profiles.
The checked three-seed pilot is retained byte-for-byte. New q=1 responses use
the analytic local self-colonization solver, the same b=1, x=0.5, raw patch
areas, hybrid distances, multi-patch exclusion and residual tolerance 1e-12.
The model source and all original pilot artifacts remain unchanged.

Replication increased first to five seeds. The predeclared diagnostic gate
checked occupancy/contrast changes above 0.05, association changes above 0.1,
or domain omission shifts above 0.25 log10 decades or a changed dispersal
endpoint. A detected change authorised the configured final step to eight.
Stage comparisons and omission checks are saved. These tolerances are screening
conventions, not tests of significance or proof of sample adequacy.

## Working q=1 parameter domain

The sample-conditional working range is **r={decision['r_min']:.8g} to
{decision['r_max']:.8g}**, with dispersal half-distance **1 to
{decision['half_distance_max']:g} cell-coordinate units**.
The separate regular survey contains 481 fixed parameter combinations:
37 quarter-decade ratio levels from .01 to 10^7 and 13 doubling half-distances
from 1 to 4096. All seven scenarios and H=.025,.2,.4,.6,.825 enter selection.
No scenario-contrast, correlation or bird-fit criterion selects the domain.

The working range retains a central90% scenario-H median transition-bracket
envelope from occupancy .95 to .05, with an outward .25-decade margin. The
upper half-distance is the first plateau onset for which the95th percentile
of shifts in the .50 median crossing is <=.05 log10 decades for at least
two successive doubling steps and all later sampled steps. Both ratio
endpoints were then evaluated at every retained half-distance on every
eligible stored H. At least95% of scenario-H median cells per distance must
reach >=.95 at the lower-r endpoint and <=.05 at the upper-r endpoint. Any
failure expands the corresponding endpoint by .25 decades; the validation
history and exact eligible-cell counts are saved. The plateau diagnostic
remains conditional on the five selection H levels.

Complete-seed omission gives log10-r lower bounds
{omission.log10_r_min.min():.3f}–{omission.log10_r_min.max():.3f}, upper bounds
{omission.log10_r_max.min():.3f}–{omission.log10_r_max.max():.3f}, and dispersal
upper endpoints {sorted(omission.half_distance_max.unique().tolist())}.
Sensitivity to plateau tolerances .025/.10, margins .125/.50 and a lower
half-distance of two is saved alongside the main decision. These numerical
bounds describe this landscape ensemble and are not biological species limits.
They define a separate sensitivity domain; no baseline domain or numerical
reference is replaced and no new parameter-space distribution is adopted.

## Results B-D at matched parameters

There are {audit['n_landscapes']} sampled landscape states,
{audit['n_excluded_landscapes']} common multi-patch exclusions and
{audit['n_valid_pairs']} valid paired profile-landscape responses.
For the unchanged B reference, the maximum absolute scenario-contrast change
is {high:.6f}; over sampled H<=.825 it is {ordinary:.6f}. The high-H results
remain conditional on the fragmented subset of landscapes and are interpreted
with the saved scenario/H exclusion counts. This is mean patch occupancy,
not occupied habitat area or a measured persistence probability.

The median absolute q=0/q=1 change over paired Result-D coefficients is
{audit['D_median_absolute_change']:.6f};
{audit['D_opposite_sign_count']} coefficients change sign with both magnitudes
above .1. Equal total scenario weights are applied after paired exclusions.
Pooling scenarios does not control for scenario membership. Missing or
numerically constant cells remain undefined. Full species-parameter
associations and all six metric associations are saved, with stage and seed
omission diagnostics. Large local changes preclude a uniform robustness claim.

Requiring response ranges above0.05 under both models retains
{ds['n_informative_paired']} paired coefficients, with median absolute model
change {ds['median_absolute_model_change']:.6f} and
{ds['opposite_sign_above_01']} sign reversals with both magnitudes above0.1.
Rounding occupancy to1e-8 changes {100*ds['rounding_fraction_over_005']:.3f}%
of finite comparable coefficients by more than0.05 and makes
{ds['rounding_newly_undefined']} formerly defined coefficients unavailable.
These numerical/response-range diagnostics accompany interpretation and do
not replace the primary coefficients or establish general metric rankings.

No bootstrap confidence intervals or bird refits were added. The larger seed
sample improves the descriptive comparison but does not establish precision
for every parameter/H cell. B ribbons show5th–95th percentiles across generated
landscapes. They describe landscape variation, not confidence in a median.

## Figure captions

B: Median equilibrium mean patch occupancy and scenario-minus-Only-Random
median contrasts under q=0/q=1, unchanged reference domain131, {n} shared
landscape seeds. Ribbons show5th–95th percentiles across landscapes. Dashed
lines denote Access-based; colours identify rho. At least three valid
multi-patch landscapes are required for each median. H decreases left to right.

C: Marginal Spearman associations of pointwise contrasts with log10-r and
log10-half-distance across the same65 profiles at each H and scenario.
Panels compare q=0/q=1; styles follow B. Profiles are fixed, with no smoothing,
profile resampling or confidence intervals. Undefined cells remain missing.

D: Equal-scenario weighted within-H metric–occupancy rank associations for
the unchanged reference, comparing q=0/q=1. {n} complete shared seeds provide
landscapes; every scenario requires at least three valid paired observations.
Colours identify the six metrics. No confidence intervals are shown.

Range survey: Median q=1 occupancy at H=.4 for half-distances1,16,256, against
r on a log axis, seven scenarios and {n} shared seeds. Styles follow B. Lines
join coarse sampled ratios; no smoothing, extrapolation or CIs are used.

Domain: Central90% scenario-H transition-bracket envelope at each sampled
half-distance, using .95 and .05 occupancy crossings, plus .50-crossing shift
diagnostics. Points are the sampled doubling distances; lines connect them
without fitting a smooth boundary. The retained interval and validated
working ratio endpoints are marked. Envelope width describes variation
between sampled scenario-H response transitions, not a confidence interval.

## Reproduction

From repository root, run final-self-colonization-extension, then
final-self-colonization-extension-review with stages validation, audit, export.
Both commands use the stable project runner. The original pilot and all
intermediate five-seed tables remain separate from this final stage.
'''
    (stage/'13_extension_report.md').write_text(text)
    land[land.replicate<=n].assign(excluded=lambda x:x.n_clusters<2).groupby(['scenario_id','H_target']).excluded.agg(['sum','count']).reset_index().to_csv(stage/'tables/13_exclusions.csv',index=False)

def domain_plot(stage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    apply_final_style();decision=json.loads((stage/'logs/13_q1_validated_domain.json').read_text())
    env=pd.DataFrame(decision['transition_envelopes']);steps=pd.DataFrame(decision['plateau_steps'])
    fig,axes=plt.subplots(1,2,figsize=figure_size(85));fig.subplots_adjust(bottom=.36,wspace=.30)
    ax=axes[0];ax.fill_between(env.half_distance,10**env.log10_r_low,10**env.log10_r_high,color='#377A9F',alpha=.2)
    ax.plot(env.half_distance,10**env.log10_r_low,color='#377A9F',marker='o',ms=3);ax.plot(env.half_distance,10**env.log10_r_high,color='#377A9F',marker='o',ms=3)
    ax.axhline(decision['r_min'],color='.35',lw=.8);ax.axhline(decision['r_max'],color='.35',lw=.8)
    ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_ylabel('Ratio, $r$ (log scale)')
    ax.text(0,1.05,'(a) Working transition domain',transform=ax.transAxes)
    ax=axes[1];ax.plot(steps.half_distance_low,steps.q95_shift_log10_r,color='#377A9F',marker='o',ms=3)
    ax.axhline(.05,color='.35',lw=.8);ax.axvline(decision['half_distance_max'],color='#E69F00',lw=1)
    ax.set_xscale('log',base=2);ax.set_ylim(bottom=0);ax.set_ylabel(r'95th-percentile shift in $\log_{10}r$')
    ax.text(0,1.05,'(b) Dispersal plateau',transform=ax.transAxes)
    fig.supxlabel(r'Dispersal half-distance, $\ell_{1/2}$ (cell-coordinate units; log scale)')
    fig.legend(handles=[Patch(facecolor='#377A9F',alpha=.2,label='Central 90% transition brackets'),Line2D([],[],color='.35',label='Working ratio bounds / tolerance'),Line2D([],[],color='#E69F00',label='Retained upper half-distance')],ncol=1,loc='lower center',frameon=False)
    export_figure(fig,stage/'figures/13_q1_parameter_domain');fig.savefig(stage/'figures/13_q1_parameter_domain.png',dpi=300,bbox_inches='tight');plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['validation','audit','export'],required=True);parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();c,d,ext,domain,land,data,out,fp=E.setup(E.DEFAULT)
    n=json.loads((out/'logs/13_compute_complete.json').read_text())['n_seeds'];stage=out/f'seeds{n:03d}'
    if args.stage=='validation':endpoint_validation(c,domain,land,data,out,n,args.workers)
    if args.stage=='audit':
        E.V.audit({**c,'replicates':list(range(1,n+1))},d,land[land.replicate<=n],stage,E.view(data,n))
        independent_domain_audit(c,domain,land,data,out,n);omission_summary(c,d,land,data,out,n);report(c,d,land,data,out,n)
    if args.stage=='export':E.V.export(c,d,stage);domain_plot(stage)

if __name__=='__main__':main()
