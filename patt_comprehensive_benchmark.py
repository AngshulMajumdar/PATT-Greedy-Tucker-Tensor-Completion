"""Reproducible CPU benchmark for budget-aware Tucker rank exchange vs baseline solvers.
All methods here are local Python implementations, NOT authors' official code.
Datasets: arbitrary-order synthetic Tucker tensors + optional scikit-learn digits.
Masks: MCAR; structured mode-0 slab exclusion.
Metrics: held-out NMSE (from noiseless truth), train NMSE, time, parameters.
"""
import os, time, json, argparse, math, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from patt_tucker_correction_experiment import orth, predict, core_fit, als_sweep, sampled_features
from patt_practical_rank_exchange import budget, core_rotate_prune, expand_from_residual

def synth(seed,shape,true_ranks,obs,mask='mcar',noise=0.002,case='synthetic'):
    rng=np.random.default_rng(seed)
    if case=='digits':
        from sklearn.datasets import load_digits
        data=load_digits().images.astype('float64') /16.
        # 12 x 12 x 8 x 8: 144 distinct digit images, fourth-order real tensor
        Y=data[rng.permutation(len(data))[:144]].reshape(12,12,8,8)
        shape=Y.shape
        truth=None
    else:
        truth=[orth(rng.normal(size=(n,r))) for n,r in zip(shape,true_ranks)]
        g=rng.normal(size=true_ranks)
        # scale entries to RMS ~1, ensuring noise has comparable interpretation across orders
        full_idx=np.array(np.unravel_index(np.arange(np.prod(shape)),shape)).T
        Y=predict(truth,g,full_idx).reshape(shape)
        Y/=np.sqrt(np.mean(Y**2))
    flat=Y.ravel(); total=flat.size
    allidx=np.arange(total)
    if mask=='mcar':
        train=rng.choice(total,size=int(total*obs),replace=False)
    elif mask=='slab':
        # Forbid the final 20% of slices along first axis; sample same number elsewhere
        forb_start=int(math.ceil(shape[0]*.80))
        eligible=allidx[np.unravel_index(allidx,shape)[0] < forb_start]
        if len(eligible)<int(total*obs): raise ValueError('requested observation density not feasible for slab mask')
        train=rng.choice(eligible,size=int(total*obs),replace=False)
    elif mask=='block':
        # Forbid a hyperrectangular block spanning ~30% of first two modes
        coords=np.unravel_index(allidx,shape)
        eligible=allidx[~((coords[0] >= int(shape[0]*.65)) & (coords[1]>=int(shape[1]*.65)))]
        if len(eligible)<int(total*obs): raise ValueError('not enough eligible entries')
        train=rng.choice(eligible,size=int(total*obs),replace=False)
    else:raise ValueError(mask)
    remain=np.setdiff1d(allidx,train,assume_unique=False)
    rng.shuffle(remain)
    nv=min(int(total*.08),len(remain)//3,2000)
    nt=min(int(total*.12),len(remain)-nv,4000)
    val=remain[:nv]; test=remain[nv:nv+nt]
    def unravel(ii):return np.array(np.unravel_index(ii,shape)).T.copy()
    yt=flat[train]+noise*rng.normal(size=len(train));yv=flat[val];yte=flat[test]
    return (yt,unravel(train)),(yv,unravel(val)),(yte,unravel(test)),truth

def evaluate(f,g,train,val,test):
    tr=np.mean((train[0]-predict(f,g,train[1]))**2) /max(np.mean(train[0]**2),1e-20)
    te=np.mean((test[0]-predict(f,g,test[1]))**2)/max(np.mean(test[0]**2),1e-20)
    ve=np.mean((val[0]-predict(f,g,val[1]))**2)/max(np.mean(val[0]**2),1e-20)
    return float(tr),float(ve),float(te)

def run_als(f,train,sweeps):
    for _ in range(sweeps):f,g,l=als_sweep(train[0],f,train[1])
    g,l=core_fit(train[0],f,train[1]);return f,g

def run_ihooi(shape,train,ranks,iterations=7):
    """Simple iterative impute-HOOI-style baseline, NOT author iHOOI code."""
    X=np.zeros(shape,float);np.put(X,np.ravel_multi_index(train[1].T,shape),train[0])
    factors=[]
    for k,r in enumerate(ranks):
        mat=np.moveaxis(X,k,0).reshape(shape[k],-1)
        Q,_,_=np.linalg.svd(mat,full_matrices=False)
        factors.append(Q[:,:r].copy())
    def compress(Z,Fs):
        q=Z
        for k,u in enumerate(Fs):q=np.tensordot(u.T,q,axes=(1,k));q=np.moveaxis(q,0,k)
        return q
    def restore(core,Fs):
        q=core
        for k,u in enumerate(Fs):q=np.tensordot(u,q,axes=(1,k));q=np.moveaxis(q,0,k)
        return q
    for it in range(iterations):
        # HOOI sweeps on imputed tensor
        for k in range(len(shape)):
            q=X
            for j,u in enumerate(factors):
                if j==k:continue
                q=np.tensordot(u.T,q,axes=(1,j));q=np.moveaxis(q,0,j)
            mat=np.moveaxis(q,k,0).reshape(shape[k],-1)
            u,_,_=np.linalg.svd(mat,full_matrices=False)
            factors[k]=u[:,:ranks[k]].copy()
        c=compress(X,factors)
        X=restore(c,factors)
        X[tuple(train[1].T)]=train[0]
    g,_=core_fit(train[0],factors,train[1]);return factors,g

def run_cp_wopt(shape,train,cp_rank,iterations=65,seed=1):
    """CP weighted-observation L-BFGS analog of published CP-WOPT; own implementation."""
    rng=np.random.default_rng(seed)
    y,idx=train
    sizes=[n*cp_rank for n in shape]
    boundaries=np.cumsum([0]+sizes)
    scale=(1/max(1,cp_rank))**(1/len(shape))
    x0=np.concatenate([(scale*rng.standard_normal((n,cp_rank))).ravel() for n in shape])
    def unpack(x):return [x[boundaries[k]:boundaries[k+1]].reshape(shape[k],cp_rank) for k in range(len(shape))]
    def fun(x):
        fac=unpack(x)
        rows=[fac[k][idx[:,k]] for k in range(len(shape))]
        pref=[np.ones((len(y),cp_rank))]
        for r in rows:pref.append(pref[-1]*r)
        suff=[None]*(len(shape)+1);suff[-1]=np.ones((len(y),cp_rank))
        for k in range(len(shape)-1,-1,-1):suff[k]=suff[k+1]*rows[k]
        residual=np.sum(pref[-1],axis=1)-y
        gradients=[]
        for k,n in enumerate(shape):
            weights=residual[:,None]*pref[k]*suff[k+1]
            grad=np.zeros((n,cp_rank))
            for j in range(cp_rank):grad[:,j]=np.bincount(idx[:,k],weights=weights[:,j],minlength=n)
            gradients.append(grad.ravel())
        return .5*np.dot(residual,residual),np.concatenate(gradients)
    result=minimize(fun,x0,jac=True,method='L-BFGS-B',options={'maxiter':iterations,'ftol':1e-10,'gtol':1e-6,'maxls':15})
    return unpack(result.x),result.nit

def cp_predict(factors,idx):
    m=len(idx);part=np.ones((m,factors[0].shape[1]))
    for k,u in enumerate(factors):part*=u[idx[:,k]]
    return part.sum(axis=1)

def run_halrtc(shape,train,iterations=30,mu=.6):
    """HaLRTC-style ADMM over nuclear norms of all mode unfoldings (own implementation)."""
    d=len(shape);X=np.full(shape,float(np.mean(train[0])));X[tuple(train[1].T)]=train[0]
    Z=[X.copy() for k in range(d)];dual=[np.zeros(shape,float) for k in range(d)]
    norm_s=float(np.sqrt(np.mean(train[0]**2)))
    alpha=1/d;mu=mu/max(norm_s,1e-8)
    for _ in range(iterations):
        for k in range(d):
            mat=np.moveaxis(X+dual[k]/mu,k,0).reshape(shape[k],-1)
            U,s,Vh=np.linalg.svd(mat,full_matrices=False)
            s=np.maximum(s-alpha/mu,0)
            Z[k]=np.moveaxis(((U*s)@Vh).reshape((shape[k],)+tuple(shape[j] for j in range(d) if j!=k)),0,k)
        X=sum((Z[k]-dual[k]/mu for k in range(d)))/d
        X[tuple(train[1].T)]=train[0]
        for k in range(d):dual[k]+=mu*(X-Z[k])
    return X

def ex_candidate(task,y,idx,factors,base_core,shape,B):
    donor,recv=task
    prune=core_rotate_prune(factors,base_core,donor)
    if prune is None:return None
    rank=[v.shape[1] for v in prune];rank[recv]+=1
    if rank[recv]>shape[recv] or budget(rank,shape)>B:return None
    pg,pl=core_fit(y,prune,idx)
    enlarged=expand_from_residual(y,idx,prune,recv,pg)
    if enlarged is None:return None
    ng,nloss=core_fit(y,enlarged,idx)
    return nloss,donor,recv,enlarged

def run_patt(init,train,val,shape,B,base_sweeps=7, exchanges=2,workers=1,shortlist=3,exchange_sweeps=2):
    f,g=run_als([u.copy() for u in init],train,base_sweeps)
    y,idx=train
    moves=[];scan_s=0.;refine_s=0.
    for z in range(exchanges):
        base_core,base_loss=core_fit(y,f,idx)
        tasks=[(i,j) for i in range(len(shape)) for j in range(len(shape)) if i!=j and f[i].shape[1]>1]
        t0=time.perf_counter()
        if workers==1:
            scores=[ex_candidate(task,y,idx,f,base_core,shape,B) for task in tasks]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                scores=list(pool.map(lambda task:ex_candidate(task,y,idx,f,base_core,shape,B),tasks))
        scores=sorted([v for v in scores if v is not None],key=lambda x:x[0])
        scan_s+=time.perf_counter()-t0
        if not scores:break
        t0=time.perf_counter()
        current_val=np.mean((val[0]-predict(f,base_core,val[1]))**2)
        choice=None; winner=None;best_val=current_val
        for nloss,donor,recv,candidate in scores[:shortlist]:
            c,cg=run_als([u.copy() for u in candidate],train,exchange_sweeps)
            trloss=.5*np.dot(y-predict(c,cg,idx),y-predict(c,cg,idx))
            vloss=np.mean((val[0]-predict(c,cg,val[1]))**2)
            if trloss<=base_loss+1e-8 and vloss<best_val-1e-10:
                best_val=vloss;winner=(c,cg);choice=(donor,recv)
        refine_s+=time.perf_counter()-t0
        if choice is None:break
        f,g=winner;moves.append(choice)
    g,_=core_fit(train[0],f,train[1])
    return f,g,moves,scan_s,refine_s

def budget_candidates(shape,start,B):
    # Non-oracle adjacent and global one-hop rank vectors with exact-cost feasibility
    d=len(shape);r=list(start);seen={tuple(r)}
    for i in range(d):
        for v in [-1,1]:
            s=r.copy();s[i]+=v
            if 1<=s[i]<=shape[i] and budget(s,shape)<=B:seen.add(tuple(s))
        for j in range(d):
            if i==j:continue
            s=r.copy();s[i]-=1;s[j]+=1
            if all(1<=ss<=nn for ss,nn in zip(s,shape)) and budget(s,shape)<=B:seen.add(tuple(s))
    return sorted(seen,key=lambda x:(sum(abs(a-b) for a,b in zip(x,r)),tuple(x)))

def test_case(config,seed,methods,workers=1,als_sweeps=7):
    name,shape,true,start,frac,mask,kind=config
    train,val,test,_=synth(seed,shape,true,frac,mask,case=kind)
    rng=np.random.default_rng(seed+91022)
    init=[orth(rng.normal(size=(n,r))) for n,r in zip(shape,start)]
    B=budget(start,shape)
    results=[]
    def record(method,elapsed,train_nmse,val_nmse,test_nmse,param,info=None):
        results.append({'case':name,'shape':shape,'order':len(shape),'rank_true':true,'rank_start':start,'mask':mask,'frac':frac,'seed':seed,'budget':B,'method':method,'train_nmse':train_nmse,'val_nmse':val_nmse,'test_nmse':test_nmse,'time_s':elapsed,'params':param,'workers':workers,'info':info or {}})
    if 'ALS' in methods or 'PATT' in methods:
        t=time.perf_counter();f,g=run_als([x.copy() for x in init],train,als_sweeps);elapsed=time.perf_counter()-t
        if 'ALS' in methods: record('Tucker-ALS',elapsed,*evaluate(f,g,train,val,test),B)
    if 'ALS-long' in methods:
        t=time.perf_counter();f,g=run_als([x.copy() for x in init],train,als_sweeps*3);record('Tucker-ALS-3x',time.perf_counter()-t,*evaluate(f,g,train,val,test),B)
    if 'ALS-multistart' in methods:
        t=time.perf_counter();candidates=[]
        for i in range(3):
            fin=[orth(np.random.default_rng(seed+10000+i).normal(size=(n,r))) for n,r in zip(shape,start)]
            f,g=run_als(fin,train,als_sweeps)
            candidates.append((evaluate(f,g,train,val,test)[1],f,g))
        _,f,g=min(candidates,key=lambda v:v[0]);record('Tucker-ALS-3starts',time.perf_counter()-t,*evaluate(f,g,train,val,test),B)
    if 'PATT' in methods:
        t=time.perf_counter();f,g,moves,ts,tr=run_patt(init,train,val,shape,B,base_sweeps=als_sweeps,exchanges=2,workers=workers,shortlist=3,exchange_sweeps=2)
        record('PATT-exchange',time.perf_counter()-t,*evaluate(f,g,train,val,test),budget(f,shape),{'moves':moves,'scan_s':ts,'refine_s':tr,'ranks':[u.shape[1] for u in f]})
    if 'rank-search' in methods:
        t=time.perf_counter();opts=budget_candidates(shape,start,B)
        # limited training budget: evaluate initial rank and up to 7 alternatives; no truth rank used
        rngrank=np.random.default_rng(seed+991)
        alternatives=opts[1:]
        ix=rngrank.permutation(len(alternatives))[:min(7,len(alternatives))]
        opts=[opts[0]]+[alternatives[q] for q in ix]
        best=None
        for rk in opts:
            ff=[orth(np.random.default_rng(seed+370+i+100*j).normal(size=(n,rr))) for j,(n,rr) in enumerate(zip(shape,rk)) for i in [0]]
            ff,gg=run_als(ff,train,als_sweeps)
            vv=evaluate(ff,gg,train,val,test)[1]
            if best is None or vv<best[0]:best=(vv,ff,gg,rk)
        _,ff,gg,rk=best
        record('Tucker-rank-search-8',time.perf_counter()-t,*evaluate(ff,gg,train,val,test),budget(rk,shape),{'ranks':rk,'num_candidates':len(opts)})
    if 'impute' in methods:
        t=time.perf_counter();f,g=run_ihooi(shape,train,start,iterations=8)
        record('Impute-HOOI',time.perf_counter()-t,*evaluate(f,g,train,val,test),B)
    if 'CP' in methods:
        t=time.perf_counter();cp_rank=min(12,max(1,B//sum(shape)))
        cf,niter=run_cp_wopt(shape,train,cp_rank,iterations=75,seed=seed+99)
        dt=time.perf_counter()-t
        pred=lambda inp:cp_predict(cf,inp[1])
        x=[float(np.mean((inp[0]-pred(inp))**2)/max(np.mean(inp[0]**2),1e-20)) for inp in [train,val,test]]
        record('CP-WOPT-style',dt,*x,sum(shape)*cp_rank,{'CP_rank':cp_rank,'iterations':niter})
    if 'HaLRTC' in methods:
        t=time.perf_counter();X=run_halrtc(shape,train,iterations=22)
        dt=time.perf_counter()-t
        xs=[float(np.mean((inp[0]-X[tuple(inp[1].T)])**2)/max(np.mean(inp[0]**2),1e-20)) for inp in [train,val,test]]
        record('HaLRTC-style',dt,*xs,int(np.prod(shape)),{'iterations':22,'dense_storage':True})
    if 'oracle' in methods and kind=='synthetic':
        t=time.perf_counter();ff=[orth(np.random.default_rng(seed+760+j).normal(size=(n,r))) for j,(n,r) in enumerate(zip(shape,true))]
        ff,gg=run_als(ff,train,als_sweeps*3)
        record('Oracle-rank-Tucker-ALS',time.perf_counter()-t,*evaluate(ff,gg,train,val,test),budget(ff,shape),{'uses_ground_truth_ranks':True})
    return results

SHAPES={
 '3S':((16,14,12),(2,3,2),(3,2,2)),
 '4S':((12,11,10,9),(2,3,2,2),(3,2,2,2)),
 '4M':((20,18,16,14),(2,3,2,3),(3,2,3,2)),
 '4L':((28,24,20,18),(3,4,3,2),(4,3,2,3)),
 '5S':((9,8,8,7,6),(2,3,2,2,2),(3,2,2,2,2)),
 '5M':((12,11,10,9,8),(3,2,2,3,2),(2,3,3,2,2)),
 '6S':((7,7,6,6,5,5),(2,3,2,2,2,2),(3,2,2,2,2,2)),
 '6M':((9,8,8,7,7,6),(2,2,3,2,2,2),(3,2,2,2,2,2)),
 'DIG':((12,12,8,8),None,(3,3,3,3)),
}
def case_spec(label,frac,mask):
    shape,true,start=SHAPES[label]
    return (label+'_'+str(frac)+'_'+mask,shape,true,start,frac,mask,'digits' if label=='DIG' else 'synthetic')

def main():
    p=argparse.ArgumentParser();p.add_argument('--shapes',default='4S,4M,5S,6S');p.add_argument('--fracs',default='0.1,0.25,0.4');p.add_argument('--masks',default='mcar,block');p.add_argument('--seeds',default='0,1');p.add_argument('--methods',default='ALS,ALS-long,ALS-multistart,PATT,rank-search,impute,CP,HaLRTC');p.add_argument('--workers',type=int,default=1);p.add_argument('--sweeps',type=int,default=7);p.add_argument('--out',default='patt_comprehensive_results.jsonl')
    args=p.parse_args();methods=args.methods.split(',');seeds=[int(v) for v in args.seeds.split(',')]
    cases=[case_spec(n,float(f),m) for n in args.shapes.split(',') for f in args.fracs.split(',') for m in args.masks.split(',')]
    with threadpool_limits(limits=1),open(args.out,'a') as fout:
        for case in cases:
            for seed in seeds:
                print('START',case[0],seed,flush=True)
                t=time.perf_counter()
                try:
                    rows=test_case(case,seed,methods,args.workers,args.sweeps)
                except Exception as exc:
                    import traceback;traceback.print_exc();fout.write(json.dumps({'case':case[0],'seed':seed,'error':repr(exc)})+'\n');fout.flush();continue
                for row in rows:fout.write(json.dumps(row)+'\n')
                fout.flush()
                print('DONE',case[0],seed,'elapsed',round(time.perf_counter()-t,2),[(r['method'],round(r['test_nmse'],4),round(r['time_s'],3)) for r in rows],flush=True)
if __name__=='__main__': main()
