"""Independent additional PATT/Tucker replication; no test information in fitting."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
import argparse, json, sys, traceback, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
S=HERE
sys.path.insert(0,str(S))
sys.path.insert(0,str(HERE))
from corrective_study import run,SHAPES
OUT=HERE/'additional_seed_trials.jsonl'
METHODS=['Tucker-ALS','Tucker-ALS-24','Tucker-HOSVD+ALS','PATT-rank-exchange']
def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',default='3,4,5');p.add_argument('--cases',default='3S,4S,5S,6S');p.add_argument('--methods',default=','.join(METHODS));p.add_argument('--out',default=str(OUT));a=p.parse_args()
 path=Path(a.out);done=set()
 if path.exists():
  for s in path.read_text().splitlines():
   try:
    x=json.loads(s);done.add((x['case'],x['seed'],x['mask'],x['rank_setup'],x['method']))
   except: pass
 with threadpool_limits(limits=1),path.open('a') as f:
  for case in a.cases.split(','):
   for mask in ['mcar','block']:
    for seed in map(int,a.seeds.split(',')):
     for rank_setup in ['misallocated','correct']:
      for method in a.methods.split(','):
       if (case,seed,mask,rank_setup,method) in done:continue
       t=time.perf_counter()
       try:
        r=run(seed,case,0.25,mask,method,rank_setup=rank_setup)
        r.update(case=case,mask=mask,seed=seed,rank_setup=rank_setup,frac=0.25,order=len(SHAPES[case][0]),error=None)
       except Exception as exc:
        r={'case':case,'mask':mask,'seed':seed,'rank_setup':rank_setup,'frac':0.25,'method':method,'error':repr(exc),'traceback':traceback.format_exc(),'total_s':time.perf_counter()-t}
       f.write(json.dumps(r,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x) if isinstance(x,np.floating) else str(x))+'\n');f.flush()
       print(f'{case} {mask} s{seed} {rank_setup} {method}: NMSE={r.get("test_nmse",float("nan")):.5g}; s={r.get("total_s",0):.2f}',flush=True)
if __name__=='__main__':main()
