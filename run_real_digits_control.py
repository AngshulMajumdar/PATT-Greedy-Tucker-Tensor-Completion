"""Observed-entry completion on reshaped scikit-learn handwritten digits (not synthetic low-rank)."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
S=HERE
sys.path.insert(0,str(S))
from patt_comprehensive_benchmark import case_spec,test_case
OUT=HERE/'real_digits_4d_trials.jsonl'

if __name__=='__main__':
 done=set()
 if OUT.exists():
  for ln in OUT.read_text().splitlines():
   try:
    v=json.loads(ln);done.add((v['case'],v['seed'],v['method']))
   except:pass
 with threadpool_limits(limits=1),OUT.open('a') as f:
  for frac in [.25,.4]:
   for mask in ['mcar','block']:
    conf=case_spec('DIG',frac,mask)
    for seed in [0,1,2]:
     t=time.perf_counter()
     try:
      rows=test_case(conf,seed,['ALS','ALS-long','PATT','rank-search','impute'],1,7)
      for r in rows:
       if (r['case'],r['seed'],r['method']) not in done:
        r['is_real_data']=True;r['dataset']='sklearn_digits_144x8x8_reshaped_12x12x8x8'
        r['pass_test']=bool(r['test_nmse']<=1e-3)
        f.write(json.dumps(r,default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n');f.flush()
      print(conf[0],seed,'elapsed_s',round(time.perf_counter()-t,2),[(a['method'],round(a['test_nmse'],5)) for a in rows],flush=True)
     except Exception as e:
      f.write(json.dumps({'case':conf[0],'seed':seed,'error':repr(e),'traceback':traceback.format_exc()})+'\n');f.flush();print('ERROR',repr(e),flush=True)
