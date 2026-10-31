"""Paired benchmarks for Python translations of the original MATLAB solvers.
No ground-truth values used for training, rank-selection, or early stopping.
"""
import os
for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[v]='1'
import json,argparse,time,sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from tram_matlab_python_port import run_tram,predict as tram_predict
from tmac_matlab_python_port import tmac,evaluate as tmac_evaluate,nmse

DIR=Path(__file__).resolve().parent
MATS=DIR

def load_trial(p):
 a=loadmat(str(p))
 shape=tuple(a['shape'].ravel().astype(int));start=tuple(a['initial_ranks'].ravel().astype(int));true=tuple(a['reference_ranks'].ravel().astype(int))
 def item(prefix):
  return (a[prefix+'_y'].ravel().astype(float),a[prefix+'_idx'].astype(int)-1)
 return shape,start,true,item('train'),item('val'),item('test')

def run_one(path,method,rank_setup='misallocated',maxiter=100):
 shape,start,true,tr,va,te=load_trial(path)
 if rank_setup=='correct':start=true
 id=path.stem
 params={'source_mat':str(path.name),'case':id.split('_')[0],'trial_id':id,'rank_setup':rank_setup,
         'true_ranks':list(true),'start_ranks':list(start),'observed_count':len(tr[0]),
         'shape':list(shape),'method':method,'status':'ok'}
 if method.startswith('TRAM'):
  if len(shape)!=3:raise ValueError('Original MATLAB TRAM_TC implemented only for 3rd-order tensors')
  if method=='TRAM-adaptive':
   # Upper rank tuple can exceed starting Tucker parameter budget;
   # record that difference explicitly.
   ceiling=tuple(max(a,b) for a,b in zip(start,true))
  elif method=='TRAM-fixed':ceiling=start
  elif method=='TRAM-oracle-rank':ceiling=true;start=true
  else:raise ValueError(method)
  r,model=run_tram(tr,va,shape,start,ceiling,seed=int(id[-1]),
                   maxiter=maxiter,maxiter_rgd=5,gradtol=0.1,delta=0.1,init='fit-core',return_model=True,adapt=method!='TRAM-fixed')
  te_pred=tram_predict(model,te[1]);params['rank_ceiling']=list(ceiling)
  params['parameter_budget_respected']=int(sum(n*r for n,r in zip(shape,model.ranks))+np.prod(model.ranks))<=int(sum(n*r for n,r in zip(shape,start))+np.prod(start))
 elif method.startswith('TMac'):
  policy=method.split('-')[-1]
  r,factors,weights=tmac(tr,va,shape,start,seed=int(id[-1]),rank_policy=policy,maxit=maxiter,
                         rank_min=[1]*len(shape),rank_max=[min(n,max(s,t)+2) for n,s,t in zip(shape,start,true)],
                         tol=1e-6,return_state=True)
  te_pred=tmac_evaluate(factors,weights,shape,te[1]);params['parameter_budget_respected']=None
 else:raise ValueError(method)
 params.update({k:v for k,v in r.items() if k not in ('method','provenance')})
 params['test_nmse']=nmse(te_pred,te[0]);params['pass_test']=params['test_nmse']<=1e-3
 params['pass_val']=params['val_nmse']<=1e-3
 params['algorithm_provenance']=r['provenance']
 return params

def main():
 a=argparse.ArgumentParser()
 a.add_argument('--methods',default='TMac-fixed,TMac-inc,TMac-dec,TRAM-adaptive,TRAM-fixed')
 a.add_argument('--rank-setups',default='misallocated,correct')
 a.add_argument('--out',default=str(DIR/'matlab_to_python_benchmark.jsonl'))
 a.add_argument('--maxiter',type=int,default=80)
 a.add_argument('--maxcases',type=int,default=0)
 args=a.parse_args()
 files=sorted(MATS.glob('*.mat'))
 if args.maxcases:files=files[:args.maxcases]
 methods=args.methods.split(',');setups=args.rank_setups.split(',')
 outpath=Path(args.out);done=set()
 if outpath.exists():
  for l in outpath.read_text().splitlines():
   try:
    j=json.loads(l);done.add((j['source_mat'],j['method'],j['rank_setup']))
   except (ValueError,KeyError):pass
 with threadpool_limits(limits=1),outpath.open('a') as f:
  for p in files:
   shape,_,_,_,_,_=load_trial(p)
   for setup in setups:
    for method in methods:
     if method.startswith('TRAM') and len(shape)!=3:continue
     key=(p.name,method,setup)
     if key in done:continue
     ts=time.perf_counter()
     try:r=run_one(p,method,setup,args.maxiter)
     except Exception as e:
      import traceback
      r={'source_mat':p.name,'method':method,'rank_setup':setup,'status':'error','error':repr(e),'traceback':traceback.format_exc(),'total_s':time.perf_counter()-ts}
     f.write(json.dumps(r,default=lambda x:float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else str(x))+'\n');f.flush()
     print(p.name,setup,method,r.get('pass_test'),f"{r.get('test_nmse',float('nan')):.5g}",f"{r.get('total_s',0):.3f}",r.get('error',''),flush=True)
if __name__=='__main__':main()
