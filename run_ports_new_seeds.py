"""Extend auditable MATLAB-source Python ports to independent seeds 3-5 at 25%."""
import os
for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[v]='1'
import sys,json,time,traceback
from pathlib import Path
import numpy as np
from scipy.io import savemat
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
SRC=HERE
PORT=HERE
sys.path.insert(0,str(SRC));sys.path.insert(0,str(PORT))
from patt_comprehensive_benchmark import SHAPES,synth
from run_matlab_ports_benchmark import run_one
MATS=HERE;MATS.mkdir(exist_ok=True)
OUT=HERE/'python_ports_additional_seeds.jsonl'

def inputs():
 for case in ['3S','4S','5S','6S']:
  shape,t,s=SHAPES[case]
  for mask in ['mcar','block']:
   for seed in [3,4,5]:
    p=MATS/f'{case}_f25_{mask}_s{seed}.mat'
    if not p.exists():
     tr,va,te,_=synth(seed,shape,t,.25,mask,noise=.002)
     savemat(str(p),{'shape':shape,'train_idx':tr[1]+1,'train_y':tr[0],
          'val_idx':va[1]+1,'val_y':va[0], 'test_idx':te[1]+1,'test_y':te[0],
          'initial_ranks':s,'reference_ranks':t,'seed':seed,'case':case,'mask':mask,'observed_fraction':.25},do_compression=True)
    yield p

def main():
 done=set()
 if OUT.exists():
  for ln in OUT.read_text().splitlines():
   try:
    r=json.loads(ln);done.add((r['source_mat'],r['method'],r['rank_setup']))
   except:pass
 with threadpool_limits(limits=1),OUT.open('a') as output:
  for p in inputs():
   for setup in ['misallocated','correct']:
    methods=['TMac-fixed','TMac-inc','TMac-dec']
    if p.name.startswith('3S'):methods+=['TRAM-adaptive','TRAM-fixed']
    for method in methods:
     if (p.name,method,setup) in done:continue
     begin=time.perf_counter()
     try:
      r=run_one(p,method,setup,120 if method.startswith('TRAM') else 100)
     except Exception as e:
      r=dict(source_mat=p.name,method=method,rank_setup=setup,status='error',error=repr(e),traceback=traceback.format_exc(),total_s=time.perf_counter()-begin)
     output.write(json.dumps(r,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n');output.flush()
     print(p.name,setup,method,r.get('test_nmse'),round(r.get('total_s',0),3),r.get('error'),flush=True)
if __name__=='__main__':main()
