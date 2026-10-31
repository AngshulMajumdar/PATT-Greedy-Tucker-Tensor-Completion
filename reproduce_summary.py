"""Audit aggregate figures in README against the shipped raw CSVs.
Run: python reproduce_summary.py
This does not rerun the stochastic algorithms; see the benchmark runner scripts.
"""
from pathlib import Path
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parent
P=pd.read_csv(R/'results_primary_25pct.csv')
assert len(P)==432 and set(P.rank_setup)=={'correct','wrong'}
assert P.error.isna().all()
S=P.assign(success=P.test_nmse<=1e-3).groupby(['rank_setup','order','method']).agg(
    trials=('success','size'),successes=('success','sum'),
    median_nmse=('test_nmse','median'),median_total_s=('total_s','median')).reset_index()
print('MAIN BENCHMARK — all measured 25%-observed trials:')
print(S.to_string(index=False,float_format=lambda x:f'{x:.6g}'))
wrong=S[S.rank_setup=='wrong']
assert wrong.query("method=='PATT-rank-exchange'").successes.sum()==32
assert wrong.query("method=='Tucker-rank-search-8'").successes.sum()==14
assert wrong.query("method=='Tucker-ALS'").successes.sum()==0

A=pd.read_csv(R/'results_actual_rank_recovery.csv');assert len(A)==48
print('\nRank allocations corrected:',int(A.rank_correct.sum()),'of',len(A))

ports=pd.read_csv(R/'results_ports_25pct.csv')
print('Source-guided port runs at 25%:',len(ports))
assert len(ports)==336
r=pd.read_csv(R/'results_real_digits.csv'); print('Real-digit trials:',sum(r.n),'successes:',sum(r.passes))
assert sum(r.n)==60 and sum(r.passes)==0

c=pd.read_csv(R/'results_certified_priority.csv');assert len(c)==48
for method,g in c.groupby('method'):
 print('Certificate-priority variant:',method,'runs',len(g),'pass',int((g.test_nmse<=1e-3).sum()),
       'median expanded core fits by order',g.groupby('order').n_expanded_core_fits.median().to_dict())

b=pd.read_csv(R/'results_post_exchange_basin.csv')
print('\nPost-exchange validation after 10 ALS sweeps:')
print(b.groupby('p').reached_target.agg(['sum','count']).to_string())

s=pd.read_csv(R/'results_spectral_stability.csv');print('Targeted bound applies in',int(s.targeted_eligible.sum()),'/',len(s))
assert len(s)==72 and int(s.targeted_eligible.sum())==70
print('All top-line assertions passed. See raw CSVs for every failed trial.')
