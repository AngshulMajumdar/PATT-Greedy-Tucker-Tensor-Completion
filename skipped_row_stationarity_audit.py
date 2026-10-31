"""Same masked row-wise Tucker ALS routine as benchmark; audit row coverage,
row Gram eigenvalues, horizontal factor gradient and skipped-row component.
The gradient is evaluated at sweep boundary, *not* misrepresented as stationarity.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,csv
import numpy as np
from numpy.linalg import norm,svd,eigvalsh
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from patt_tucker_correction_experiment import orth,predict,core_fit,als_sweep,mode_coefficients,design
from threadpoolctl import threadpool_limits

def trial(seed,p,shape=(8,9,7,7),r=(2,2,2,2)):
    rng=np.random.default_rng(seed)
    tru=[orth(rng.normal(size=(n,z))) for n,z in zip(shape,r)]
    gtrue=rng.normal(size=r)
    idx=np.array(np.unravel_index(rng.choice(np.prod(shape),round(np.prod(shape)*p),replace=False),shape)).T
    y=predict(tru,gtrue,idx)+.001*rng.normal(size=len(idx))
    U=[orth(A+.06*rng.normal(size=A.shape)) for A in tru]
    core,_=core_fit(y,U,idx)
    # executed row Gram statistics under exactly the same mode sweeps as ALS
    ff=[a.copy() for a in U]; gg=core.copy()
    skip_pre=[];obs_counts=[];grams=[]
    for k,M in enumerate(ff):
        T,_=mode_coefficients(ff,gg,idx,k)
        B=M.copy()
        s=np.zeros(M.shape[0],dtype=bool)
        for a in range(M.shape[0]):
            ix=idx[:,k]==a
            obs_counts.append(int(ix.sum()))
            if ix.sum()<M.shape[1]:
                s[a]=True
                continue
            ev=eigvalsh(T[ix].T@T[ix]); grams.append(ev[0]);B[a]=np.linalg.lstsq(T[ix],y[ix],rcond=None)[0]
        skip_pre.append(s)
        Q,R=np.linalg.qr(B,mode='reduced')
        gg=np.moveaxis(np.tensordot(R,gg,axes=(1,k)),0,k)
        ff[k]=Q
        gg,_=core_fit(y,ff,idx)
    # check faithful against actual routine
    f2,g2,loss2=als_sweep(y,U,idx)
    gg,loss=core_fit(y,ff,idx)
    if abs(loss-loss2)>1e-7:raise RuntimeError(('not same code',seed,p,loss,loss2))
    residual=predict(ff,gg,idx)-y
    core_grad=design(ff,idx).T@residual
    sqskip=sqexec=sqall=0.
    for k,M in enumerate(ff):
        T,_=mode_coefficients(ff,gg,idx,k)
        G=np.zeros_like(M)
        np.add.at(G,idx[:,k],residual[:,None]*T)
        H=G-M@(M.T@G)   # horizontal gradient, invariant to orthogonal Tucker gauge
        norms=(H**2).sum(axis=1)
        sqskip+=float(norms[skip_pre[k]].sum())
        sqexec+=float(norms[~skip_pre[k]].sum())
        sqall+=float(norms.sum())
    # regularity and data-driven diagnostics, not a convergence guarantee
    gaps=[]
    for k in range(len(r)):
        sv=svd(np.moveaxis(gg,k,0).reshape(r[k],-1),compute_uv=False)
        gaps.append(float(sv[-1]))
    return dict(seed=seed,p=p,nobserved=len(idx),updated_rows=sum(len(x)-x.sum() for x in skip_pre),
        skipped_rows=sum(x.sum() for x in skip_pre),min_executed_row_gram=min(grams,default=0.),
        median_executed_row_gram=float(np.median(grams)) if grams else 0.,
        projected_gradient_norm=np.sqrt(sqall+float(norm(core_grad)**2)),
        skipped_horizontal_gradient_norm=np.sqrt(sqskip),
        executed_horizontal_gradient_norm=np.sqrt(sqexec),
        core_gradient_norm=norm(core_grad),min_core_unfolding_sigma=min(gaps),
        train_loss=loss)

if __name__=='__main__':
 out=[]
 with threadpool_limits(limits=1):
    for p in [.002,.005,.01,.10]:
        for z in range(12):
            a=trial(9200+z,p)
            out.append(a)
            print('p',p,'seed',z,'skip',a['skipped_rows'],'rskip',round(a['skipped_horizontal_gradient_norm'],7),flush=True)
 p=Path(__file__).with_name('skipped_row_stationarity_results.csv')
 with p.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
