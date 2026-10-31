"""Matched eight-config rank-search control, no oracle information used in selection.
Only the rank budget of the initial Tucker model enters candidate generation.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
import sys,time,json,traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
S=HERE
sys.path.insert(0,str(S))
from patt_comprehensive_benchmark import SHAPES,synth,run_als,budget_candidates
from patt_practical_rank_exchange import budget
from patt_tucker_correction_experiment import orth,predict
OUT=HERE/'rank_search_8_additional.jsonl';TARGET=1e-3

def nmse(f,g,dat):
 y,idx=dat;return float(np.mean((predict(f,g,idx)-y)**2)/max(1e-20,float(np.mean(y*y))))

def case_trial(case,mask,seed):
 shape,real,start=SHAPES[case]
 tr,va,te,_=synth(seed,shape,real,0.25,mask,noise=.002)
 B=budget(start,shape);opts=budget_candidates(shape,start,B)
 alt=opts[1:]
 rng=np.random.default_rng(seed+991)
 ix=rng.permutation(len(alt))[:min(7,len(alt))]
 optset=[opts[0]]+[alt[i] for i in ix]
 began=time.perf_counter();history=[];best=None
 for ci,ranks in enumerate(optset):
  factors=[orth(np.random.default_rng(seed+370+100*j).normal(size=(n,r))) for j,(n,r) in enumerate(zip(shape,ranks))]
  f,g=run_als(factors,tr,6)
  vn=nmse(f,g,va);history.append({'candidate':ci,'ranks':ranks,'val_nmse':vn,'time_s':time.perf_counter()-began})
  if best is None or vn<best[0]:best=(vn,f,g,ranks)
  # Test values are never accessed until all selection is done.
  if vn<=TARGET:break
 v,f,g,r=best
 return dict(case=case,order=len(shape),frac=.25,mask=mask,seed=seed,rank_setup='misallocated',method='Tucker-rank-search-8',
             test_nmse=nmse(f,g,te),val_nmse=nmse(f,g,va),train_nmse=nmse(f,g,tr),total_s=time.perf_counter()-began,
             first_val_target_s=next((h['time_s'] for h in history if h['val_nmse']<=TARGET),None),
             ranks=list(r),params=budget(r,shape),budget=B,initial_ranks=list(start),evaluated_candidates=len(history),
             history=history,pass_val=bool(v<=TARGET),pass_test=bool(nmse(f,g,te)<=TARGET),error=None)
if __name__=='__main__':
 done=set()
 if OUT.exists():
  for l in OUT.read_text().splitlines():
   try:
    a=json.loads(l);done.add((a['case'],a['mask'],a['seed']))
   except Exception:pass
 with threadpool_limits(limits=1),OUT.open('a') as out:
  for case in ['3S','4S','5S','6S']:
   for mask in ['mcar','block']:
    for seed in range(6):
     if (case,mask,seed) in done:continue
     t=time.perf_counter()
     try:r=case_trial(case,mask,seed)
     except Exception as e:r={'case':case,'mask':mask,'seed':seed,'method':'Tucker-rank-search-8','error':repr(e),'traceback':traceback.format_exc(),'total_s':time.perf_counter()-t}
     out.write(json.dumps(r,default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n');out.flush()
     print(case,mask,seed,r.get('test_nmse',float('nan')),r.get('evaluated_candidates'),f'{r["total_s"]:.2f}',r.get('error'),flush=True)
