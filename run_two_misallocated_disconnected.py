"""Exploratory two-misplaced-rank stress test, same original PATT implementation."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,time,json
from pathlib import Path
S=Path(__file__).resolve().parent
sys.path.insert(0,str(S))
import corrective_study as C
from threadpoolctl import threadpool_limits
# Two ranks misallocated from mode 1 to mode 2; budget true=136, intermediate=146, starting=148.
C.SHAPES['4S2']=((12,11,10,9),(2,4,2,2),(4,2,2,2))
O=Path(__file__).with_name('two_misallocated_disconnected.jsonl')
with O.open('w') as f,threadpool_limits(limits=1):
 for mask in ('mcar','block'):
  for seed in (31,32,33,34):
   for meth in ('PATT-rank-exchange','Tucker-ALS','Tucker-ALS-24','Tucker-HOSVD+ALS'):
    t=time.perf_counter();
    try:
     r=C.run(seed,'4S2',.25,mask,meth,exchange_rounds=2,rank_setup='misallocated')
     d={'method':meth,'mask':mask,'seed':seed,'test_nmse':r['test_nmse'],'success':int(r['pass_test']), 'ranks':r['ranks'],'exchanges':r.get('exchanges',[]),'runtime':r['total_s'],'error':None}
    except Exception as e:
     d={'method':meth,'mask':mask,'seed':seed,'error':repr(e)}
    f.write(json.dumps(d)+'\n');f.flush();print(mask,seed,meth,d.get('test_nmse'),d.get('exchanges'),flush=True)
