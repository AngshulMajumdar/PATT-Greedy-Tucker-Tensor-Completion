"""Matched, one-exchange comparison. Baseline = original PATT exhaustive core-screen,
shortlist/refine/validation. Alternative = cert-priority candidate evaluation; it
changes selection and is *not* claimed to be the same method. Uses actual post-prune
residual score, and counts expanded-core LS solves. No ground truth enters selection.
"""
import os
for e in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[e]='1'
import time,csv,sys
import numpy as np
from numpy.linalg import norm,svd
from pathlib import Path
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from patt_tucker_correction_experiment import core_fit,predict,als_sweep,orth,sampled_features
from patt_practical_rank_exchange import seed_misallocated,budget,core_rotate_prune,expand_from_residual

def candidates(y,idx,factors,base_core,base_loss,shape,B):
    out=[]
    for donor in range(len(shape)):
        if factors[donor].shape[1]<=1: continue
        reduced=core_rotate_prune(factors,base_core,donor)
        cg,fd=core_fit(y,reduced,idx)
        loss=fd-base_loss
        e=y-predict(reduced,cg,idx)
        for j in range(len(shape)):
            if j==donor: continue
            rr=[x.shape[1] for x in reduced];rr[j]+=1
            if rr[j]>shape[j] or budget(rr,shape)>B: continue
            ph=sampled_features(reduced,idx,exclude=j)
            H=np.zeros((shape[j],ph.shape[1]),dtype=float)
            np.add.at(H,idx[:,j],e[:,None]*ph)
            U=reduced[j];H-=U@(U.T@H)
            if np.max(np.abs(H))<1e-11: continue
            us,s,_=svd(H,full_matrices=False)
            v=us[:,0];v-=U@(U.T@v)
            if norm(v)<1e-10:continue
            v=v/norm(v)
            cand=[a.copy() for a in reduced];cand[j]=np.column_stack((cand[j],v))
            lb=.5*s[0]**2-loss
            out.append(dict(donor=donor,receiver=j,lb=lb,candidate=cand))
    return out

def run_one(seed,shape,rt,ri,observed):
    y,idx,yv,iv,yt,it,initial=seed_misallocated(seed,shape,rt,ri,observed=observed,init_noise=.15)
    B=budget(ri,shape)
    U=[a.copy() for a in initial]
    for _ in range(5): U,_,_=als_sweep(y,U,idx)
    G,F=core_fit(y,U,idx)
    result=[]
    for method in ['original_full_screen','certified_priority']:
        t0=time.perf_counter()
        C=candidates(y,idx,U,G,F,shape,B)
        vals=[]; solves=0
        if method=='original_full_screen':
            for d in C:
                cg,loss=core_fit(y,d['candidate'],idx);solves+=1
                vals.append((loss,d))
            ordered=[d for _,d in sorted(vals,key=lambda x:x[0])[:4]]
        else:
            ordered=sorted(C,key=lambda d:d['lb'],reverse=True)
        incumbent_valid=.5*norm(yv-predict(U,G,iv))**2
        bestval=incumbent_valid;bf=U;bg=G;bc=F;choice='none';evaluated=0
        for d in ordered:
            if method=='certified_priority':
                # A negative lower bound is inconclusive, so never prune using it.
                cg,loss=core_fit(y,d['candidate'],idx);solves+=1
            cf=[a.copy() for a in d['candidate']]
            for _ in range(2): cf,cg,loss=als_sweep(y,cf,idx)
            cg,loss=core_fit(y,cf,idx)
            val=.5*norm(yv-predict(cf,cg,iv))**2
            evaluated+=1
            if loss<=F+1e-8 and val <bestval-1e-9:
                bestval=val;bf=cf;bg=cg;bc=loss;choice=f"{d['donor']}->{d['receiver']}"
                if method=='certified_priority':
                    break # Early-accept modification: differs from original minimum-validation shortlist
        tst=.5*norm(yt-predict(bf,bg,it))**2
        den=.5*norm(yt)**2
        result.append(dict(seed=seed,order=len(shape),p=observed,method=method,
            n_candidates=len(C),n_expanded_core_fits=solves,n_refined=evaluated,
            n_positive_cert=sum(d['lb']>0 for d in C),
            best_candidate_cert=max([d['lb'] for d in C],default=0.),
            accepted=choice!='none',choice=choice,train_loss=bc,
            val_loss=bestval,test_nmse=tst/max(den,1e-30),wall_s=time.perf_counter()-t0))
    return result

if __name__=='__main__':
    cases=[((11,10,9,8),(2,3,2,2),(3,2,2,2)),((8,8,7,7,6),(2,3,2,2,2),(3,2,2,2,2))]
    out=[]
    with threadpool_limits(limits=1):
        for shape,rt,ri in cases:
            for p in [.25,.40]:
                for seed in range(8000,8006):
                    z=run_one(seed,shape,rt,ri,p)
                    out+=z
                    print(len(shape),p,seed,[(a['method'],a['n_expanded_core_fits'],round(a['wall_s'],3),round(a['test_nmse'],5)) for a in z],flush=True)
    outname=Path(__file__).with_name('certified_priority_results.csv')
    with open(outname,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
