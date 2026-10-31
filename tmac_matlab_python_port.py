"""Python translation of the TMac.m alternating-unfolding-factor algorithm.

MATLAB source: https://github.com/lymch/ARE/blob/main/Tucker/TMac.m
Original algorithm: Xu, Hao, Yin, Su, Parallel matrix factorization for
low-rank tensor completion, Inverse Problems and Imaging 9 (2015).

This is a closely source-mapped Python port (not author-run code). Translated:
  - mode-n factorization, alternating X/Y updates, weighted completion;
  - rank-inc and rank-dec helpers, alpha adaptation, stopping;
  - normalize X,Y by observed-entry scale;
  - no dense true tensor leakage into algorithm.
Bug/compatibility corrections: nstall initialized (MATLAB source doesn't do it);
rank decreasing handles singular-value edge cases; validation early stopping
is an optional benchmark wrapper, not part of original TMac code.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import time


def unfold(a,k):
    return np.moveaxis(a,k,0).reshape(a.shape[k],-1)

def fold(x,shape,k):
    return np.moveaxis(x.reshape((shape[k],)+tuple(n for i,n in enumerate(shape) if i!=k)),0,k)

def evaluate(factors,weights,shape,idx):
    # predictions may be materialized for small benchmark; MATLAB holds M dense
    return sum(float(weights[k])*fold(x@y,shape,k)[tuple(idx.T)] for k,(x,y) in enumerate(factors))

def nmse(pred,ref):
    return float(np.mean((pred-ref)**2)/max(float(np.mean(ref**2)),1e-20))

def tmac(train,val,shape,rank_init,*,seed=0,rank_policy='fixed',rank_min=None,rank_max=None,
         maxit=50,tol=1e-4,alpha_adj=False,stop_val_nmse=1e-3,return_state=False):
    if rank_policy not in ('fixed','inc','dec'):raise ValueError(rank_policy)
    rng=np.random.default_rng(200001+seed)
    y,idx=train;d=len(shape);N=int(np.prod(shape));m=len(y)
    ranks=np.asarray(rank_init,dtype=int).copy()
    rmin=np.ones(d,dtype=int) if rank_min is None else np.asarray(rank_min,dtype=int)
    rmax=np.minimum(np.asarray(shape,dtype=int),ranks+3) if rank_max is None else np.asarray(rank_max,dtype=int)
    weights=np.full(d,1/d)
    scale=max(float(np.linalg.norm(y))*np.sqrt(N/m),1e-10)
    M=np.zeros(shape)
    M[tuple(idx.T)]=y
    factors=[]
    for k,(n,r) in enumerate(zip(shape,ranks)):
        nc=N//n
        X=rng.normal(size=(n,r));Y=rng.normal(size=(r,nc))
        X*=scale**(n/(n+nc))/max(float(np.linalg.norm(X)),1e-30)
        Y*=scale**(nc/(n+nc))/max(float(np.linalg.norm(Y)),1e-30)
        factors.append((X,Y))
    residual0=np.array([np.linalg.norm(fold(x@z,shape,k)[tuple(idx.T)]-y) for k,(x,z) in enumerate(factors)])
    previous=float(np.sum(residual0**2))
    stall=0;start=time.perf_counter();history=[];rank_events=[]
    def record(step):
        preds=evaluate(factors,weights,shape,val[1])
        e=nmse(preds,val[0])
        history.append({'step':step,'val_nmse':e,'time_s':time.perf_counter()-start,'ranks':ranks.tolist()})
        return e
    va=record(0)
    reason='maxit'
    if va<=stop_val_nmse:reason='target'
    for it in range(1,maxit+1):
        if reason=='target':break
        for k,(X,Y) in enumerate(factors):
            Mk=unfold(M,k)
            X=Mk@Y.T
            Gram=X.T@X
            Y=np.linalg.pinv(Gram,rcond=1e-11)@X.T@Mk
            factors[k]=(X,Y)
            if rank_policy=='dec' and ranks[k]>rmin[k]:
                # faithfully translate rank_dec_adaptive using Gram X^TX
                sval=np.linalg.svd(Gram,compute_uv=False)
                start_idx=max(0,int(rmin[k])-1)
                drops=sval[start_idx:-1]/np.maximum(sval[start_idx+1:],1e-30)
                if len(drops):
                    j=int(np.argmax(drops));dmx=drops[j]
                    den=float(drops.sum()-dmx)
                    rd=(ranks[k]-rmin[k])*dmx/max(den,1e-30)
                    if rd>10 and j+int(rmin[k])<ranks[k]:
                        newrank=j+int(rmin[k])
                        qx,rx=np.linalg.qr(X,mode='reduced')
                        qy,ry=np.linalg.qr(Y.T,mode='reduced')
                        su,s,sv=np.linalg.svd(rx@ry.T,full_matrices=False)
                        X=qx@su[:,:newrank]*s[:newrank]
                        Y=(qy@sv.T[:,:newrank]).T
                        factors[k]=(X,Y)
                        ranks[k]=newrank
                        rank_events.append({'iter':it,'mode':k,'action':'dec','ranks':ranks.tolist()})
        pred=[fold(x@z,shape,k) for x,z in factors]
        residual=np.asarray([np.linalg.norm(a[tuple(idx.T)]-y) for a in pred])
        M=sum(weights[k]*pred[k] for k in range(d))
        M[tuple(idx.T)]=y
        total=float(np.sum(residual**2))
        rel_change=np.abs(1-residual/np.maximum(residual0,1e-15))
        if rank_policy=='inc':
            for k,(X,Y) in enumerate(factors):
                if ranks[k]>=rmax[k] or rel_change[k]>=max(1e-2,10*tol):continue
                # Rank increment as in rank_inc_adaptive.m:
                # grow Y's row subspace by a random orthogonal direction.
                q,_=np.linalg.qr(Y.T,mode='reduced')
                v=rng.normal(size=Y.shape[1]);v-=q@(q.T@v)
                v/=max(float(np.linalg.norm(v)),1e-30)
                newY=np.vstack([Y,v]);newX=unfold(M,k)@newY.T
                factors[k]=(newX,newY);ranks[k]+=1
                rank_events.append({'iter':it,'mode':k,'action':'inc','ranks':ranks.tolist()})
        if alpha_adj:
            weights=(1/np.maximum(residual**2,1e-30))
            weights/=weights.sum()
        va=record(it)
        if va<=stop_val_nmse:reason='target';break
        rel=abs(total-previous)/(previous+1)
        stall=stall+1 if rel<tol else 0
        if stall>=3 or (weights@residual)/max(float(np.linalg.norm(y)),1e-30)<tol:
            reason='converged';break
        residual0=residual;previous=total
    runtime=time.perf_counter()-start
    res={'method':'TMac-MATLAB-logic-Python-port-'+rank_policy,'ranks':ranks.tolist(),
         'train_nmse':nmse(evaluate(factors,weights,shape,idx),y),
         'val_nmse':nmse(evaluate(factors,weights,shape,val[1]),val[0]),
         'first_val_target_s':next((a['time_s'] for a in history if a['val_nmse']<=stop_val_nmse),None),
         'total_s':runtime,'iterations':it if reason!='target' or len(history)>1 else 0,
         'history':history,'rank_events':rank_events,'reason':reason,
         'provenance':'source-mapped independent Python port of TMac.m; not authors MATLAB executable'}
    return (res,factors,weights) if return_state else res
