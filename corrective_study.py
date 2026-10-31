"""PATT within-Tucker corrective study: measured CPU-only masked completion.
Published-method-inspired solvers are author-independent local implementations.
Train samples generate candidates; validation selects/halts; test is unseen until final evaluation.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import json,time,argparse,math,sys
from pathlib import Path
from functools import lru_cache
import numpy as np
from threadpoolctl import threadpool_limits
from patt_tucker_correction_experiment import orth,predict,core_fit,als_sweep, finite_candidate
from patt_practical_rank_exchange import budget,exchange_search, core_rotate_prune, expand_from_residual
from patt_comprehensive_benchmark import SHAPES,synth,run_ihooi,budget_candidates
from patt_tucker_wopt_baseline import fit_wopt

TARGET=1e-3

def err(f,g,obs):
    y,idx=obs
    return float(np.mean((y-predict(f,g,idx))**2)/max(np.mean(y*y),1e-20))

def model(f,g,train,val,test,start,history,label,**kwargs):
    vals=dict(method=label,train_nmse=err(f,g,train),val_nmse=err(f,g,val),test_nmse=err(f,g,test),
              total_s=time.perf_counter()-start,ranks=[x.shape[1] for x in f],
              params=budget(f,[x.shape[0] for x in f]),history=history,**kwargs)
    vals['pass_test']=vals['test_nmse']<=TARGET
    vals['pass_val']=vals['val_nmse']<=TARGET
    checkpoints=[x['time_s'] for x in history if x['val_nmse']<=TARGET]
    vals['first_val_target_s']=min(checkpoints) if checkpoints else None
    return vals

def checkpoint(f,g,val,start,history,label):
    e=err(f,g,val)
    history.append(dict(step=label,val_nmse=e,time_s=time.perf_counter()-start))
    return e

def spectral_factors(train,shape,ranks):
    # Zero-filled HOSVD from TRAIN ONLY; this is merely a proposal, not a completion claim.
    Y=np.zeros(shape,float)
    y,idx=train
    Y[tuple(idx.T)]=y
    out=[]
    for k,(n,r) in enumerate(zip(shape,ranks)):
        Z=np.moveaxis(Y,k,0).reshape(n,-1)
        uu,_,_=np.linalg.svd(Z,full_matrices=False)
        out.append(uu[:,:r].copy())
    return out

def replacement_candidates(f, proposal, *, pairs=True, all_modes=True):
    d=len(f)
    combos=[(k,) for k in range(d)]
    if pairs: combos += [(i,j) for i in range(d) for j in range(i+1,d)]
    if all_modes: combos.append(tuple(range(d)))
    out=[];seen=set()
    for combo in combos:
        if combo in seen:continue
        seen.add(combo)
        cand=[x.copy() for x in f]
        for k in combo:cand[k]=proposal[k].copy()
        out.append((combo,cand))
    return out

def try_reseed(f,g,train,val,start,history, max_trials=3, with_finite=False):
    shape=[x.shape[0] for x in f]
    ranks=[x.shape[1] for x in f]
    try:
        spectral=spectral_factors(train,shape,ranks)
    except (np.linalg.LinAlgError,ValueError):return f,g,False,0
    base_tr=core_fit(train[0],f,train[1])[1]
    base_val=err(f,g,val)
    scored=[]; count=0
    for combo,cand in replacement_candidates(f,spectral):
        cand_g,cl=core_fit(train[0],cand,train[1]);count+=1
        if not np.isfinite(cl):continue
        # All-mode candidate can have higher immediate residual before fitting. Require train objective descent.
        if cl < base_tr-1e-9:
            scored.append((cl,combo,cand,cand_g))
    scored.sort(key=lambda item:item[0])
    bestf,bestg=f,g;bestval=base_val;besttrain=base_tr;choice=None
    # Screening is train-only; validation is used for acceptance after 1 ALS sweep.
    for _,combo,cand,gg in scored[:max_trials]:
        cf=[u.copy() for u in cand]
        cf,cg,newtrain=als_sweep(train[0],cf,train[1])
        ev=err(cf,cg,val)
        if newtrain<besttrain-1e-9 and ev<bestval-1e-9:
            bestf,bestg=cf,cg;bestval=ev;besttrain=newtrain;choice=combo
    if choice is None:return f,g,False,count
    checkpoint(bestf,bestg,val,start,history,'joint_replace_'+str(choice))
    return bestf,bestg,choice,count

def enhanced_exchange_search(f,g,train,val,B, *, shortlist=4, guide=True, sweeps=1):
    """Greedy rank exchange with rank-corrective, data-driven receiver proposals.
    All candidate core fits use training entries; validation used only in accept/reject.
    Besides conditional residual features, tests train-only sparse-unfolding directions.
    """
    y,idx=train; shape=[u.shape[0] for u in f];d=len(f)
    base_loss=core_fit(y,f,idx)[1];base_val=err(f,g,val)
    spectral=spectral_factors(train,shape,[min(u.shape[1]+1,u.shape[0]) for u in f]) if guide else None
    options=[]
    for donor in range(d):
        if f[donor].shape[1]<=1:continue
        pruned=core_rotate_prune(f,g,donor)
        if pruned is None:continue
        pc,_=core_fit(y,pruned,idx)
        for recv in range(d):
            if recv==donor or pruned[recv].shape[1]>=shape[recv]:continue
            rk=[u.shape[1] for u in pruned];rk[recv]+=1
            if budget(rk,shape)>B:continue
            proposals=[]
            residual=expand_from_residual(y,idx,pruned,recv,pc)
            if residual is not None:proposals.append(('conditional',residual))
            if guide:
                U=pruned[recv]
                # Provide globally informed receiver candidates rather than
                # conditioning exclusively on all other current mode factors.
                for a in range(min(3,spectral[recv].shape[1])):
                    v=spectral[recv][:,a].copy()
                    v-=U@(U.T@v)
                    norm=np.linalg.norm(v)
                    if norm<1e-8:continue
                    v/=norm
                    cand=[u.copy() for u in pruned]
                    cand[recv]=np.column_stack([U,v])
                    proposals.append((f'spectral{a}',cand))
            for typ,cand in proposals:
                cg,cl=core_fit(y,cand,idx)
                options.append((cl,donor,recv,typ,cand))
    if not options:return f,g,None,0
    options.sort(key=lambda z:z[0])
    best=f;bg=g;best_val=base_val;best_tr=base_loss;which=None
    # Evaluate more than only the numerically best screened candidate.
    for _,donor,recv,typ,cand in options[:shortlist]:
        cf=[u.copy() for u in cand]
        for _ in range(sweeps):
            cf,cg,cl=als_sweep(y,cf,idx)
        cg,cl=core_fit(y,cf,idx)
        vv=err(cf,cg,val)
        if cl<best_tr-1e-8 and vv<best_val-1e-9:
            best=cf;bg=cg;best_tr=cl;best_val=vv;which=[donor,recv,typ]
    return best,bg,which,len(options)

def run(seed,case,frac,mask,method,init_variant='random', max_als=6, exchange_rounds=2,rank_setup='misallocated'):
    shape,rt,rstart=SHAPES[case]
    if rank_setup=='correct':rstart=rt
    train,val,test,_=synth(seed,shape,rt,frac,mask,noise=.002)
    rng=np.random.default_rng(seed+91022)
    if init_variant=='random':
        init=[orth(rng.normal(size=(n,r))) for n,r in zip(shape,rstart)]
    elif init_variant.startswith('wrong'):
        # Construct more hostile subspaces for controlled correction: incorrect all modes,
        # ranks remain as in rstart. This does not reveal the truth, unless used for a separate stress-only split.
        init=[orth(rng.normal(size=(n,r))) for n,r in zip(shape,rstart)]
    else:raise ValueError(init_variant)
    B=budget(rstart,shape)
    start=time.perf_counter();h=[];f=[x.copy() for x in init]
    exchanges=[]; replacements=[]; screened=0
    if method in ('Tucker-HOSVD+ALS','PATT-guided-HOSVD','PATT-HOSVD-exchange'):
        f=spectral_factors(train,shape,rstart)
    if method=='Impute-HOOI':
        f,g=run_ihooi(shape,train,rstart,iterations=8)
        checkpoint(f,g,val,start,h,'ihooi_end')
        return model(f,g,train,val,test,start,h,method)
    if method=='Tucker-WOPT-local':
        f,g,it,conv=fit_wopt(shape,train,rstart,seed=seed,iterations=100)
        checkpoint(f,g,val,start,h,'wopt_end')
        return model(f,g,train,val,test,start,h,method,iterations=it)
    sweeps = 24 if method=='Tucker-ALS-24' else max_als
    if method=='Tucker-HOSVD+ALS':sweeps=8
    for i in range(sweeps):
        f,g,_=als_sweep(train[0],f,train[1])
        if checkpoint(f,g,val,start,h,'als_%d'%(i+1))<=TARGET:break
    if method=='PATT-dual-seed' and h[-1]['val_nmse']>TARGET:
        fz=spectral_factors(train,shape,rstart)
        for i in range(max_als):
            fz,gz,_=als_sweep(train[0],fz,train[1])
            # HOSVD candidate selection is always train+val monitored; no test feedback.
            # Do not treat a screened but rejected alternative as an accepted iterate.
            hv=err(fz,gz,val)
            if hv<=TARGET:break
        if err(fz,gz,val)<err(f,g,val) and core_fit(train[0],fz,train[1])[1]<core_fit(train[0],f,train[1])[1]:
            f,g=fz,gz
            checkpoint(f,g,val,start,h,'accepted_alt_hosvd')
    if h[-1]['val_nmse']>TARGET and method in ('PATT-rank-exchange','PATT-swap-before','PATT-swap-after','PATT-swap-only','PATT-guided-exchange','PATT-guided-HOSVD','PATT-HOSVD-exchange','PATT-dual-seed'):
        if method in ('PATT-swap-before','PATT-swap-only'):
            f,g,ch,n=try_reseed(f,g,train,val,start,h,max_trials=3)
            screened+=n
            if ch is not False:replacements.append(list(ch))
        if method != 'PATT-swap-only' and (not h or h[-1]['val_nmse']>TARGET):
            for t in range(exchange_rounds):
                g,oldtrain=core_fit(train[0],f,train[1])
                if method in ('PATT-guided-exchange','PATT-guided-HOSVD','PATT-HOSVD-exchange','PATT-dual-seed'):
                    nf,ng,choice,nc=enhanced_exchange_search(f,g,train,val,B,shortlist=5)
                    screened+=nc
                else:
                    nf,ng,nloss,choice=exchange_search(train[0],train[1],f,g,oldtrain,shape,B,
                                                       sweeps=1,shortlist=3,validation=val)
                if choice is None:break
                f,g=nf,ng;exchanges.append(choice)
                if checkpoint(f,g,val,start,h,'exchange_%d'%(t+1))<=TARGET:break
        if method=='PATT-swap-after' and (not h or h[-1]['val_nmse']>TARGET):
            f,g,ch,n=try_reseed(f,g,train,val,start,h,max_trials=3)
            screened+=n
            if ch is not False:replacements.append(list(ch))
    g,_=core_fit(train[0],f,train[1])
    return model(f,g,train,val,test,start,h,method,exchanges=exchanges,
                 replacements=replacements,replacement_candidates_screened=screened)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',default='3S,4S')
    p.add_argument('--fracs',default='.1,.25,.4')
    p.add_argument('--masks',default='mcar,block')
    p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--methods',default='Tucker-ALS,Tucker-ALS-24,Tucker-HOSVD+ALS,PATT-rank-exchange,PATT-swap-before,PATT-swap-after,PATT-swap-only,Tucker-WOPT-local,Impute-HOOI')
    p.add_argument('--out',default='corrective_results.jsonl')
    p.add_argument('--rank-setup',default='misallocated',choices=['misallocated','correct'])
    p.add_argument('--skip-existing',action='store_true')
    args=p.parse_args()
    done=set()
    if args.skip_existing and Path(args.out).exists():
        for line in Path(args.out).read_text().splitlines():
            try:
                v=json.loads(line);done.add((v['case'],v['frac'],v['mask'],v['seed'],v['method']))
            except (ValueError,KeyError): pass
    with threadpool_limits(limits=1),open(args.out,'a') as output:
        for case in args.cases.split(','):
            for frac in map(float,args.fracs.split(',')):
                for mask in args.masks.split(','):
                    for seed in map(int,args.seeds.split(',')):
                        for meth in args.methods.split(','):
                            if (case,frac,mask,seed,meth) in done:continue
                            begin=time.perf_counter()
                            try:
                                res=run(seed,case,frac,mask,meth,rank_setup=args.rank_setup)
                                res.update(case=case,seed=seed,frac=frac,mask=mask,order=len(SHAPES[case][0]),
                                           true_ranks=SHAPES[case][1],start_ranks=(SHAPES[case][1] if args.rank_setup=='correct' else SHAPES[case][2]),rank_setup=args.rank_setup,error=None)
                            except Exception as exc:
                                import traceback
                                res=dict(case=case,seed=seed,frac=frac,mask=mask,method=meth,
                                         total_s=time.perf_counter()-begin,error=str(exc),traceback=traceback.format_exc())
                            output.write(json.dumps(res)+'\n');output.flush()
                            print(case,frac,mask,seed,meth,round(res.get('test_nmse',float('nan')),6),
                                  round(res['total_s'],3),'ERR' if res['error'] else '',flush=True)
if __name__=='__main__':main()

# End module.
