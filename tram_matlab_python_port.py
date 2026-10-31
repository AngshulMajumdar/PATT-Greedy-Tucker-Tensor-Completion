"""Source-guided NumPy translation of the MATLAB TRAM_TC algorithm (Gao/Peng/Yuan).

Original source: https://github.com/JimmyPeng1998/TRAM
Files studied: TRAM_TC.m, calcFunction.m, calcGradient.m, calcProjection.m,
calcInitial.m, retraction.m, rankIncreasing.m, rankDecreasing.m,
computeNormalBasis.m, innerProduct.m, hosvd.m.

The MATLAB implementation is order-3, and so is this port. Core operations are
NumPy rather than Tensor Toolbox/MEX. Correctness changes versus MATLAB:
  * safeguarded line search rejects numerical objective increases;
  * never increases rank beyond the specified rank ceiling;
  * explicit validation NMSE early stop for the benchmark (no test leakage);
  * optional rank-cap enforcement on adaptive moves.
These changes and the lack of MATLAB comparison mean this is a Python port,
NOT an execution of or numerically certified equivalent to author code.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import time
import numpy as np
from dataclasses import dataclass
from scipy.linalg import svd

@dataclass
class Tucker:
    core: np.ndarray
    factors: list[np.ndarray]

    @property
    def ranks(self):
        return tuple(self.core.shape)


def ttm(a:np.ndarray,u:np.ndarray,mode:int):
    return np.moveaxis(np.tensordot(u,a,axes=(1,mode)),0,mode)

def unfold(a:np.ndarray,mode:int):
    return np.moveaxis(a,mode,0).reshape(a.shape[mode],-1)

def predict(x:Tucker,idx:np.ndarray):
    m=idx.shape[0]
    res=np.ones((m,1))
    for k,u in enumerate(x.factors):
        res=(res[:,:,None]*u[idx[:,k],None,:]).reshape(m,-1)
    return res@x.core.ravel()

def hosvd(a:np.ndarray,ranks):
    factors=[]
    for k,r in enumerate(ranks):
        q=np.linalg.svd(unfold(a,k),full_matrices=False)[0][:,:r]
        if q.shape[1]<r:raise ValueError('Invalid rank for HOSVD')
        factors.append(q)
    c=a
    for k,q in enumerate(factors):c=ttm(c,q.T,k)
    return Tucker(c,factors)

def orth_init(shape,ranks,seed=0,core_scale=1.0):
    rng=np.random.default_rng(seed)
    factors=[np.linalg.qr(rng.normal(size=(n,r)),mode='reduced')[0] for n,r in zip(shape,ranks)]
    return Tucker(core_scale*rng.normal(size=ranks),factors)

def core_from_sampled(y,idx,factors):
    # Same masked least-squares core fitting as Tucker-ALS, used for a matched
    # optional stable initial core; unlike author test script this isn't random.
    m=len(y);a=np.ones((m,1))
    for k,u in enumerate(factors):a=(a[:,:,None]*u[idx[:,k],None,:]).reshape(m,-1)
    g=np.linalg.lstsq(a,y,rcond=1e-10)[0]
    return g.reshape(tuple(u.shape[1] for u in factors))

def tangent_gradient(x:Tucker,idx:np.ndarray,errs:np.ndarray):
    """Port of calcProjection: orthogonal projection of sparse gradient onto tangent.
    For G_(k) shape r_k x R_{-k}, the differential dU_k is
      (I-U_kU_k^T) gradY x_{l != k} U_l^T times pinv(G_(k)).
    """
    r=x.ranks;d=len(r);m=len(idx)
    tangent_core=np.zeros(r)
    # accumulator Y contracted across *all* U's
    phi=np.ones((m,1))
    for k,u in enumerate(x.factors):
        phi=(phi[:,:,None]*u[idx[:,k],None,:]).reshape(m,-1)
    tangent_core=(phi.T@errs).reshape(r)
    tangent_factors=[]
    for k,u in enumerate(x.factors):
        other=np.ones((m,1))
        for ell,v in enumerate(x.factors):
            if ell==k:continue
            other=(other[:,:,None]*v[idx[:,ell],None,:]).reshape(m,-1)
        # Use np.add.at to accumulate by row index
        a=np.zeros((u.shape[0],other.shape[1]))
        np.add.at(a,idx[:,k],errs[:,None]*other)
        corek=unfold(x.core,k)
        raw=a@np.linalg.pinv(corek,rcond=1e-10)
        tangent_factors.append(raw-u@(u.T@raw))
    return tangent_core,tangent_factors

def tangent_norm_sq(x:Tucker,dc,du):
    val=float(np.sum(dc*dc))
    for k,v in enumerate(du):
        b=unfold(x.core,k)
        val+=float(np.sum((v.T@v)*(b@b.T)))
    return val

def tangent_observed(x:Tucker,idx:np.ndarray,dc,du):
    """The exact first-order directional prediction on observed entries."""
    out=predict(Tucker(dc,x.factors),idx)
    for k in range(len(x.factors)):
        newf=list(x.factors)
        newf[k]=du[k]
        out+=predict(Tucker(x.core,newf),idx)
    return out

def tucker_retract(x:Tucker,dc,du,alpha:float):
    """Direct translation of MATLAB retraction.m, generalized tensor indices."""
    r=x.ranks;d=len(r)
    if not np.isfinite(alpha):raise ValueError('invalid step')
    qs=[];rs=[]
    for u,v in zip(x.factors,du):
        q,rr=np.linalg.qr(np.concatenate([u,v],axis=1),mode='reduced')
        qs.append(q);rs.append(rr)
    s=np.zeros(tuple(2*v for v in r))
    lo=tuple(slice(0,v) for v in r)
    s[lo]=x.core+alpha*dc
    for k in range(d):
        ss=[slice(0,v) for v in r]
        ss[k]=slice(r[k],2*r[k])
        s[tuple(ss)]=alpha*x.core
    for k,rr in enumerate(rs):s=ttm(s,rr,k)
    small=hosvd(s,r)
    return Tucker(small.core,[q@v for q,v in zip(qs,small.factors)])

def normal_rank_addition(x:Tucker,idx,errs,rmax,seed,mode='random'):
    """Original TRAM rankIncreasing: one simultaneous normal-vector addition to every mode."""
    r=x.ranks
    if any(ri>=hi or ri>=ui.shape[0] for ri,hi,ui in zip(r,rmax,x.factors)):
        return None
    rng=np.random.default_rng(seed)
    vectors=[]
    for k,u in enumerate(x.factors):
        # Source basis=random uses QR([U, randn])
        v=rng.normal(size=(u.shape[0],1))
        v-=u@(u.T@v)
        v/=max(float(np.linalg.norm(v)),1e-30)
        vectors.append(v)
    cand=Tucker(np.ones((1,)*len(r)),vectors)
    z=predict(cand,idx)
    v=float(errs@z);zz=float(z@z)
    if zz<1e-24:return None
    newr=tuple(t+1 for t in r)
    newcore=np.zeros(newr)
    newcore[tuple(slice(0,t) for t in r)]=x.core
    alpha=-v/zz
    newcore[tuple(t for t in r)]=alpha
    out=Tucker(newcore,[np.column_stack([u,v]) for u,v in zip(x.factors,vectors)])
    return out,abs(v),zz

def rank_decrease(x:Tucker,delta:float=.1):
    """Translation of rankDecreasing.m: threshold the core singular values."""
    new=[]
    for k,r in enumerate(x.ranks):
        s=np.linalg.svd(unfold(x.core,k),compute_uv=False)
        if not len(s) or s[0]<1e-15: new.append(1);continue
        i=np.where(s<delta*s[0])[0]
        new.append(max(1,int(i[0])) if len(i) else r)
    if all(a==b for a,b in zip(new,x.ranks)):return None
    return hosvd(x.core,new),new

def rank_decrease_tensor(x:Tucker,delta:float=.1):
    obj=rank_decrease(x,delta)
    if obj is None:return None
    small,_=obj
    return Tucker(small.core,[u@v for u,v in zip(x.factors,small.factors)])

def _nmse(x:Tucker,data):
    y,idx=data
    return float(np.mean((predict(x,idx)-y)**2)/max(float(np.mean(y*y)),1e-20))

def run_tram(train,val,shape,r_init,r_ceiling,*,seed=0,maxiter=100,maxiter_rgd=5,
             delta=.10,gradtol=.1,stop_nmse=1e-3,objective_guard=True,
             init='random',return_model=False,adapt=True,normaltol=.1):
    """TRAM_TC (3D) with source-mapped rank-increase/decrease and Riemannian steps.
    Gradient threshold starts at 0.1 in MATLAB; numerical scale is normalized by
    observed signal RMS so that benchmark's RMS~1 does not change the meaning.
    """
    if len(shape)!=3:raise ValueError('MATLAB TRAM_TC is implemented for order three only')
    if any(a>b for a,b in zip(r_init,r_ceiling)):raise ValueError('initial rank exceeds ceiling')
    y,idx=train
    rng=np.random.default_rng(10001+seed)
    x=orth_init(shape,r_init,seed+55)
    # Initialize with masked LS core, optional; MATLAB starts random tensor
    if init=='fit-core':x.core=core_from_sampled(y,idx,x.factors)
    if init not in ('fit-core','random'):raise ValueError('init')
    grad_threshold=gradtol
    started=time.perf_counter()
    h=[];moves=[];stop='maxiter';n=0
    valerr=_nmse(x,val)
    h.append({'iter':0,'val_nmse':valerr,'elapsed_s':time.perf_counter()-started,'ranks':list(x.ranks)})
    while n<maxiter:
        inner_converged=False
        for j in range(maxiter_rgd):
            if n>=maxiter:break
            pred=predict(x,idx);res=pred-y
            dc,du=tangent_gradient(x,idx,res)
            ng=np.sqrt(tangent_norm_sq(x,dc,du))
            if ng<grad_threshold and j>0:
                inner_converged=True
                break
            pdir=tangent_observed(x,idx,dc,du)
            denom=float(pdir@pdir)
            if denom<=1e-25:
                inner_converged=True
                break
            alpha=-float(res@pdir)/denom
            before=float(res@res)
            accepted=False
            for _ in range(12):
                trial=tucker_retract(x,dc,du,alpha)
                rr=predict(trial,idx)-y
                if np.isfinite(rr).all() and (not objective_guard or float(rr@rr)<before-1e-10):
                    accepted=True
                    break
                alpha*=.5
            if not accepted:
                inner_converged=True
                break
            x=trial;n+=1
            valerr=_nmse(x,val)
            h.append({'iter':n,'val_nmse':valerr,'elapsed_s':time.perf_counter()-started,'ranks':list(x.ranks)})
            if valerr<=stop_nmse:
                stop='validation_target';break
        if stop=='validation_target':break
        if n>=maxiter:break
        # Original TRAM: if inner RGD converged, consider rank increasing;
        # else consider rank decreasing based on core singular values.
        if not adapt:
            if inner_converged:
                stop='fixed_rank_stagnation';break
            continue
        if inner_converged:
            res=predict(x,idx)-y
            proposed=normal_rank_addition(x,idx,res,r_ceiling,seed+1000+n)
            if proposed is not None:
                candidate,gain,denom=proposed
                if gain <= max(ng,1e-30)*normaltol:
                    grad_threshold *= .5
                    continue
                if (not objective_guard or np.sum((predict(candidate,idx)-y)**2)<np.sum(res**2)-1e-10):
                    x=candidate;moves.append({'iter':n,'action':'increase_all','ranks':list(x.ranks)})
                    grad_threshold=gradtol
                    h.append({'iter':n,'val_nmse':_nmse(x,val),'elapsed_s':time.perf_counter()-started,'ranks':list(x.ranks)})
                else:grad_threshold*=.5
            else:
                if grad_threshold<1e-8:stop='no_feasible_increase';break
                grad_threshold*=.5
        else:
            candidate=rank_decrease_tensor(x,delta=delta)
            if candidate is not None:
                # MATLAB rankDecreasing accepts a threshold-driven truncation
                # even when training loss temporarily increases.
                x=candidate;moves.append({'iter':n,'action':'decrease','ranks':list(x.ranks)})
                grad_threshold=gradtol
                h.append({'iter':n,'val_nmse':_nmse(x,val),'elapsed_s':time.perf_counter()-started,'ranks':list(x.ranks)})
        if len(h)>=5 and abs(h[-1]['val_nmse']-h[-2]['val_nmse'])<1e-10 and inner_converged:
            stop='stagnation';break
        if n==0 and inner_converged and not moves:stop='stationarity';break
    elapsed=time.perf_counter()-started
    out={'method':'TRAM-MATLAB-logic-Python-port','ranks':list(x.ranks),
         'train_nmse':_nmse(x,train),'val_nmse':_nmse(x,val),
         'total_s':elapsed,'first_val_target_s':next((t['elapsed_s'] for t in h if t['val_nmse']<=stop_nmse),None),
         'iterations':n,'history':h,'moves':moves,'stop_reason':stop,
         'provenance':'source-guided TRAM_TC, different numerical kernels/safeguards; MATLAB code not executed'}
    return (out,x) if return_model else out
