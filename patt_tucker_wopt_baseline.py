"""Own weighted Tucker nonlinear least squares baseline inspired by Tucker-WOPT.
This is NOT a faithful implementation of the original Grassmann optimizer.
"""
import os,sys,time,json,argparse
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from patt_comprehensive_benchmark import SHAPES,synth,case_spec,budget
from patt_tucker_correction_experiment import orth,predict,core_fit,sampled_features

def fit_wopt(shape, train, ranks, seed=0, iterations=130):
    rng=np.random.default_rng(seed+91022)
    factors=[orth(rng.normal(size=(n,r))) for n,r in zip(shape,ranks)]
    core,_=core_fit(train[0],factors,train[1]);m=len(train[0]);d=len(shape)
    sizes=[n*r for n,r in zip(shape,ranks)];cuts=np.cumsum([0]+sizes)
    core_shape=tuple(ranks); y,idx=train
    x0=np.concatenate([x.ravel() for x in factors]+[core.ravel()])
    def fun(x):
        f=[x[cuts[k]:cuts[k+1]].reshape(shape[k],ranks[k]) for k in range(d)]
        g=x[cuts[-1]:].reshape(core_shape)
        D=sampled_features(f,idx)
        residual=D@g.ravel()-y
        gradg=D.T@residual
        gradf=[]
        for k in range(d):
            phi=sampled_features(f,idx,exclude=k)
            gk=np.moveaxis(g,k,0).reshape(ranks[k],-1)
            t=phi@gk.T
            grad=np.column_stack([np.bincount(idx[:,k],weights=residual*t[:,j],minlength=shape[k]) for j in range(ranks[k])])
            gradf.append(grad.ravel())
        loss=.5*np.dot(residual,residual)
        return loss,np.concatenate(gradf+[gradg.ravel()])
    res=minimize(fun,x0,jac=True,method='L-BFGS-B',options={'maxiter':iterations,'ftol':1e-10,'gtol':1e-7,'maxls':20})
    f=[res.x[cuts[k]:cuts[k+1]].reshape(shape[k],ranks[k]) for k in range(d)]
    g=res.x[cuts[-1]:].reshape(core_shape)
    return f,g,res.nit,res.success

def main():
    p=argparse.ArgumentParser();p.add_argument('--shapes',default='4S,4M,5S,5M,6S');p.add_argument('--fracs',default='0.1,0.25,0.4');p.add_argument('--masks',default='mcar,block');p.add_argument('--seeds',default='0');p.add_argument('--iter',type=int,default=130);p.add_argument('--out',default='patt_wopt_results.jsonl');args=p.parse_args()
    with threadpool_limits(limits=1),open(args.out,'a') as out:
     for name in args.shapes.split(','):
      for frac in map(float,args.fracs.split(',')):
       for mask in args.masks.split(','):
        case,shape,true,start,_,_,kind=case_spec(name,frac,mask)
        for seed in map(int,args.seeds.split(',')):
         train,val,test,_=synth(seed,shape,true,frac,mask,case=kind)
         t=time.perf_counter();f,g,nit,success=fit_wopt(shape,train,start,seed,args.iter)
         elapsed=time.perf_counter()-t
         errs=[float(np.mean((y-predict(f,g,idx))**2)/max(np.mean(y*y),1e-20)) for (y,idx) in [train,val,test]]
         row={'case':case,'shape':shape,'order':len(shape),'rank_true':true,'rank_start':start,'mask':mask,'frac':frac,'seed':seed,'budget':budget(start,shape),'method':'Tucker-WOPT-style','train_nmse':errs[0],'val_nmse':errs[1],'test_nmse':errs[2],'time_s':elapsed,'params':budget(start,shape),'workers':1,'info':{'iterations':nit,'optimizer':'L-BFGS-B','not_official_Grassmann_solver':True,'success':bool(success)}}
         out.write(json.dumps(row)+'\n');out.flush()
         print('DONE',case,seed,'error',round(errs[2],5),'seconds',round(elapsed,3),'iters',nit,flush=True)
if __name__=='__main__':main()
