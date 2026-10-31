"""CPU-only general-order Tucker masked completion: ALS + directional rank exchanges.
Research prototype. Rank exchanges are verified by reconstruction error after refinement.
No GPU or dynamic programming. Not intended to be a production-scale implementation.
"""
import argparse, time, json
import numpy as np
from threadpoolctl import threadpool_limits
from patt_tucker_correction_experiment import orth, predict, core_fit, als_sweep, sampled_features


def budget(factors_or_ranks, shape):
    ranks=[x.shape[1] if hasattr(x,'shape') else int(x) for x in factors_or_ranks]
    return int(sum(n*r for n,r in zip(shape,ranks))+np.prod(ranks))


def core_rotate_prune(factors, core, mode):
    """Delete smallest-energy Tucker core mode direction after an orthogonal gauge rotation."""
    r=list(core.shape)
    if r[mode]<=1: return None
    unfolding=np.moveaxis(core,mode,0).reshape(r[mode],-1)
    U,s,_=np.linalg.svd(unfolding,full_matrices=False)
    out=[v.copy() for v in factors]
    out[mode]=out[mode]@U[:,:-1]
    return out


def expand_from_residual(y, idx, factors, mode, core=None, seed=0):
    """Residual-feature spectral candidate for adding a direction in one mode."""
    m=len(y); n=factors[mode].shape[0]
    if n<=factors[mode].shape[1]:return None
    if core is None:core,_=core_fit(y,factors,idx)
    residual=y-predict(factors,core,idx)
    phi=sampled_features(factors,idx,exclude=mode)
    B=np.zeros((n,phi.shape[1]),dtype=float)
    np.add.at(B,idx[:,mode], residual[:,None]*phi)
    U=factors[mode]
    B-=U@(U.T@B)
    # leading vector from matrix-free implicit B B^T: for prototype dense small B
    if np.max(np.abs(B))<1e-11:return None
    uu,ss,_=np.linalg.svd(B, full_matrices=False)
    v=uu[:,0]; v-=U@(U.T@v)
    norm=np.linalg.norm(v)
    if norm<1e-10:return None
    v/=norm
    out=[u.copy() for u in factors]
    out[mode]=np.column_stack((out[mode],v))
    return out


def seed_misallocated(seed,shape,true_ranks,start_ranks,observed=.35,noise=0.002,init_noise=.15):
    rng=np.random.default_rng(seed)
    trueU=[orth(rng.normal(size=(n,r))) for n,r in zip(shape,true_ranks)]
    truecore=rng.normal(size=true_ranks)
    total=int(np.prod(shape))
    pos=rng.permutation(total)
    m=int(observed*total)
    itrain=np.array(np.unravel_index(pos[:m],shape)).T
    ival=np.array(np.unravel_index(pos[m:m+int(.12*total)],shape)).T
    itest=np.array(np.unravel_index(pos[m+int(.12*total):m+int(.24*total)],shape)).T
    ytrain=predict(trueU,truecore,itrain)+noise*rng.normal(size=len(itrain))
    yval=predict(trueU,truecore,ival)
    ytest=predict(trueU,truecore,itest)
    init=[]
    for k,(n,r,rt) in enumerate(zip(shape,start_ranks,true_ranks)):
        n_good=min(r,rt)
        # mixed correct basis with independent noise to make less artificial
        A=trueU[k][:,:n_good] + init_noise*rng.normal(size=(n,n_good))
        if r>n_good:A=np.column_stack((A,rng.normal(size=(n,r-n_good))))
        init.append(orth(A))
    return (ytrain,itrain,yval,ival,ytest,itest,init)


def evaluate(y,idx,factors):
    core,train=core_fit(y,factors,idx)
    return core,train


def exchange_search(y,idx,factors,base_core,base_loss,shape,B,sweeps=2,shortlist=6,validation=None):
    """One rank exchange: cut one mode and expand another; refine and accept best train gain."""
    d=len(factors)
    all_candidates=[]
    # Parallelizable cheap screening; here we implement serial to establish correctness.
    for donor in range(d):
        if factors[donor].shape[1]<=1:continue
        pruned=core_rotate_prune(factors,base_core,donor)
        if pruned is None:continue
        # re-fit to current observations after deletion
        pruned_core,pruned_loss=core_fit(y,pruned,idx)
        for recv in range(d):
            if recv==donor:continue
            ranks=[v.shape[1] for v in pruned]
            ranks[recv]+=1
            if budget(ranks,shape)>B: continue
            if ranks[recv]>shape[recv]:continue
            enlarged=expand_from_residual(y,idx,pruned,recv,pruned_core)
            if enlarged is None:continue
            # screened by residual decrease after exact core re-fit; no ALS yet
            new_core,lnew=core_fit(y,enlarged,idx)
            all_candidates.append((lnew,donor,recv,enlarged))
    if not all_candidates:return factors,base_core,base_loss, None
    all_candidates.sort(key=lambda x:x[0])
    best_loss=base_loss; best_factors=factors; best_core=base_core; choice=None
    best_score=(.5*float(np.linalg.norm(validation[0]-predict(factors,base_core,validation[1]))**2) if validation is not None else base_loss)
    for _,donor,recv,cand in all_candidates[:shortlist]:
        cf=[x.copy() for x in cand]
        for it in range(sweeps):
            cf,cg,cl=als_sweep(y,cf,idx)
        cg,cl=core_fit(y,cf,idx)
        score=(.5*float(np.linalg.norm(validation[0]-predict(cf,cg,validation[1]))**2) if validation is not None else cl)
        if score<best_score-1e-9 and cl<=base_loss+1e-8:
            best_score=score;best_loss=cl;best_factors=cf;best_core=cg;choice=[donor,recv]
    return best_factors,best_core,best_loss,choice


def trial(seed,shape,true_ranks,start_ranks,sweeps=5,exchange_iters=3, observed=.35, init_noise=.15):
    y,idx,yv,iv,yt,it,init=seed_misallocated(seed,shape,true_ranks,start_ranks,observed,init_noise=init_noise)
    B=budget(start_ranks,shape)
    base=[x.copy() for x in init]
    for t in range(sweeps):
        base,core,loss=als_sweep(y,base,idx)
    core,loss=core_fit(y,base,idx)
    base_test=.5*np.linalg.norm(yt-predict(base,core,it))**2
    base_val=.5*np.linalg.norm(yv-predict(base,core,iv))**2
    start=time.perf_counter()
    f=[x.copy() for x in base]; choices=[]
    for t in range(exchange_iters):
        g,fl=core_fit(y,f,idx)
        nf,ng,nl,which=exchange_search(y,idx,f,g,fl,shape,B,sweeps=2,shortlist=4,validation=(yv,iv))
        if which is None:break
        f=nf;choices.append(which)
    core,train=core_fit(y,f,idx)
    tim=time.perf_counter()-start
    exchange_test=.5*np.linalg.norm(yt-predict(f,core,it))**2
    exchange_val=.5*np.linalg.norm(yv-predict(f,core,iv))**2
    return {'seed':seed,'B':B,'start_ranks':list(start_ranks),'true_ranks':list(true_ranks),
      'als_train':loss,'als_validation':base_val,'als_test':base_test,
      'exchange_train':train,'exchange_validation':exchange_val,'exchange_test':exchange_test,
      'exchange_s':tim,'exchanges':choices,'final_ranks':[x.shape[1] for x in f]}


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--seeds',type=int,default=3);pa.add_argument('--sweeps',type=int,default=5);pa.add_argument('--exchanges',type=int,default=2);pa.add_argument('--observed',type=float,default=.35);pa.add_argument('--init-noise',type=float,default=.15)
    args=pa.parse_args()
    cases=[((11,10,9,8),(2,3,2,2),(3,2,2,2)),
           ((8,8,7,7,6),(2,3,2,2,2),(3,2,2,2,2))]
    with threadpool_limits(limits=1):
      for shape,rt,ri in cases:
        records=[]
        for seed in range(args.seeds):
            t0=time.perf_counter();r=trial(seed,shape,rt,ri,args.sweeps,args.exchanges, observed=args.observed,init_noise=args.init_noise)
            records.append(r);r['elapsed_s']=time.perf_counter()-t0
            print(json.dumps(r),flush=True)
        print('SUMMARY',shape, 'relative test change', np.median([r['exchange_test']/r['als_test'] for r in records]),
              'relative validation change',np.median([r['exchange_validation']/r['als_validation'] for r in records]),flush=True)
if __name__=='__main__':main()
