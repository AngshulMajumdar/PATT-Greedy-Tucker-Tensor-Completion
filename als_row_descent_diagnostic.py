"""Audit *executed* row-wise ALS updates used in PATT: compare measured decrease
against exact normal-equation identity. The equality is before the optional core
refit; QR gauge steps preserve the tensor and cannot affect that observed loss.
"""
import os
for v in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[v]='1'
import sys,csv
import numpy as np
from numpy.linalg import norm
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
from patt_tucker_correction_experiment import core_fit,mode_coefficients,predict,als_sweep,orth

def one(seed,frac,shape=(8,9,7,7),rr=(2,2,2,2)):
 rng=np.random.default_rng(seed);Utrue=[orth(rng.normal(size=(n,r))) for n,r in zip(shape,rr)];Gtrue=rng.normal(size=rr)
 N=np.prod(shape);flat=rng.choice(N,int(N*frac),replace=False)
 idx=np.array(np.unravel_index(flat,shape)).T
 y=predict(Utrue,Gtrue,idx)+rng.normal(scale=.001,size=len(flat))
 U=[orth(Utrue[k]+.05*rng.normal(size=Utrue[k].shape)) for k in range(len(shape))]
 f=[u.copy() for u in U];core,base=core_fit(y,f,idx)
 sum_exact=0.;sum_lower=0.;updated=0;skipped=0;gauge_maxdiff=0.
 for k,M in enumerate(f):
  t,_=mode_coefficients(f,core,idx,k)
  newM=M.copy()
  pre=.5*norm(y-predict(f,core,idx))**2
  for row in range(M.shape[0]):
   mask=idx[:,k]==row
   if mask.sum()<M.shape[1]:skipped+=1;continue
   T=t[mask];yr=y[mask];old=M[row]
   res=T@old-yr;grad=T.T@res
   gram=T.T@T
   new=np.linalg.lstsq(T,yr,rcond=None)[0]
   reduction=.5*(norm(res)**2-norm(T@new-yr)**2)
   cert=.5*grad@(np.linalg.pinv(gram)@grad)
   if abs(reduction-cert)>1e-7*max(1.,abs(reduction)):
    raise AssertionError((seed,frac,k,row,reduction,cert))
   sum_exact+=reduction
   lam=np.linalg.eigvalsh(gram)[-1]
   if lam>0:sum_lower+=.5*float(grad@grad)/lam
   updated+=1;newM[row]=new
  Q,R=np.linalg.qr(newM,mode='reduced')
  core_before=np.moveaxis(np.tensordot(R,core,axes=(1,k)),0,k)
  f[k]=Q
  afterQ=.5*norm(y-predict(f,core_before,idx))**2
  pre_newf=f.copy();pre_newf[k]=newM
  old_unnormalized=.5*norm(y-predict(pre_newf,core,idx))**2
  gauge_maxdiff=max(gauge_maxdiff,abs(old_unnormalized-afterQ))
  core,_=core_fit(y,f,idx)
 ffin,loss=core_fit(y,f,idx)
 f2,_,l2=als_sweep(y,U,idx)
 if abs(loss-l2)>1e-7:
  raise AssertionError(('mismatch actual ALS',seed,loss,l2))
 actual=base-loss
 if actual+1e-7<sum_exact or sum_exact+1e-7<sum_lower:
  raise AssertionError(('descent violated',actual,sum_exact,sum_lower))
 return dict(seed=seed,frac=frac,initial_loss=base,final_loss=loss,
             actual_sweep_gain=actual,exact_row_gain=sum_exact,row_gradient_lower_bound=sum_lower,
             updated_rows=updated,skipped_rows=skipped,max_qr_objective_deviation=gauge_maxdiff)
if __name__=='__main__':
 out=[one(4000+i,frac) for frac in [.1,.25,.4] for i in range(12)]
 path=str(__import__('pathlib').Path(__file__).resolve().parent/'als_row_descent_diagnostics.csv')
 with open(path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
 for frac in [.1,.25,.4]:
  z=[x for x in out if x['frac']==frac]
  print(frac,'verified',len(z),'skipped_rows',sum(x['skipped_rows'] for x in z),'min gain',min(x['actual_sweep_gain'] for x in z),'max QR dev',max(x['max_qr_objective_deviation'] for x in z), 'median gain ratio',np.median([x['actual_sweep_gain']/x['exact_row_gain'] for x in z]))
