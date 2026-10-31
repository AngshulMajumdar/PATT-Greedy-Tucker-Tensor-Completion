"""Reproducible CPU experiment for general-order masked Tucker subspace correction.

Finite replacement is exact conditional least-squares over the core, given other factors.
This is a proof-of-concept for correction, NOT a complete adaptive-rank algorithm.
"""
import argparse, json, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits


def orth(X):
    Q, R = np.linalg.qr(X, mode='reduced')
    return Q


def sampled_features(factors, indices, exclude=None):
    m = len(indices)
    f = np.ones((m, 1))
    for k, U in enumerate(factors):
        if k == exclude:
            continue
        f = (f[:, :, None] * U[indices[:, k], None, :]).reshape(m, -1)
    return f


def design(factors, indices):
    return sampled_features(factors, indices)


def core_fit(y, factors, indices):
    A = design(factors, indices)
    g, *_ = np.linalg.lstsq(A, y, rcond=None)
    residual = y - A @ g
    return g.reshape(tuple(U.shape[1] for U in factors)), .5 * float(residual @ residual)


def predict(factors, core, indices):
    return design(factors, indices) @ core.ravel()


def mode_coefficients(factors, core, indices, k):
    phi = sampled_features(factors, indices, exclude=k)
    core_k = np.moveaxis(core, k, 0).reshape(factors[k].shape[1], -1)
    return phi @ core_k.T, phi


def continuous_candidate(y, factors, core, indices, k, j):
    e = y - predict(factors, core, indices)
    t, _ = mode_coefficients(factors, core, indices, k)
    n = factors[k].shape[0]
    a = np.bincount(indices[:, k], weights=e*t[:, j], minlength=n)
    U = factors[k]
    p = a - U@(U.T@a)
    norm = np.linalg.norm(p)
    if norm < 1e-12:
        return None
    w = p/norm
    v = U[indices[:, k], j]*t[:, j]
    z = w[indices[:, k]]*t[:, j]
    def loss(th):
        dx = (np.cos(th)-1)*v + np.sin(th)*z
        r = e - dx
        return .5*float(r@r)
    result = minimize_scalar(loss, method='bounded', bounds=(-np.pi/2, np.pi/2), options={'xatol':1e-9})
    if loss(0)-result.fun < 1e-11:
        return None
    newf = [u.copy() for u in factors]
    newf[k][:, j] = np.cos(result.x)*U[:, j] + np.sin(result.x)*w
    return {'factors':newf, 'predicted_loss':result.fun, 'gain':loss(0)-result.fun,
            'gradient_norm':float(norm), 'mode':k, 'column':j, 'theta':float(result.x)}


def finite_candidate(y, factors, core, indices, k, j, top=2):
    r = [U.shape[1] for U in factors]
    A = design(factors, indices)
    labels = np.indices(r).reshape(len(r), -1)[k]
    C = A[:, labels != j]
    Q, _ = np.linalg.qr(C, mode='reduced')
    rC = y-Q@(Q.T@y)
    phi = sampled_features(factors, indices, exclude=k)
    n = factors[k].shape[0]
    B = np.zeros((n,phi.shape[1]))
    np.add.at(B, indices[:,k], rC[:,None]*phi)
    Uminus = np.delete(factors[k], j, axis=1)
    B -= Uminus@(Uminus.T@B)
    left, _, _ = np.linalg.svd(B, full_matrices=False)
    def fitted_loss(u):
        W = u[indices[:,k],None] * phi
        Z = W-Q@(Q.T@W)
        a, *_ = np.linalg.lstsq(Z,rC,rcond=None)
        res = rC-Z@a
        return .5*float(res@res)
    baseline = fitted_loss(factors[k][:,j])
    candidates=[]
    for h in range(min(top,left.shape[1])):
        u = left[:,h]
        # Any component of u orthogonal to Uminus is admissible.
        u -= Uminus@(Uminus.T@u)
        nu = np.linalg.norm(u)
        if nu<1e-12: continue
        u /= nu
        fval=fitted_loss(u)
        candidates.append((fval,u.copy()))
    if not candidates:
        return None
    fval,u=min(candidates,key=lambda z:z[0])
    if baseline-fval < 1e-11:
        return None
    newf=[v.copy() for v in factors]
    newf[k][:,j]=u
    return {'factors':newf, 'predicted_loss':fval, 'gain':baseline-fval,
            'mode':k, 'column':j, 'old_loss':baseline}


def one_round(y, factors, indices, method='finite', top=2, workers=1):
    core, old_loss=core_fit(y, factors, indices)
    start=time.perf_counter()
    jobs=[(k,j) for k,U in enumerate(factors) for j in range(U.shape[1])]
    def evaluate(job):
        k,j=job
        return (finite_candidate(y,factors,core,indices,k,j,top) if method=='finite'
                else continuous_candidate(y,factors,core,indices,k,j))
    if workers>1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            candidates=[c for c in ex.map(evaluate,jobs) if c is not None]
    else:
        candidates=[c for job in jobs if (c:=evaluate(job)) is not None]
    scan_time=time.perf_counter()-start
    if not candidates:
        return factors,core,old_loss,0.,scan_time,None
    best=max(candidates,key=lambda c:c['gain'])
    core_new, loss_new=core_fit(y,best['factors'],indices)
    # The full core refit should agree with the exact conditional finite-replacement score.
    if method=='finite' and abs(loss_new-best['predicted_loss'])>1e-6*max(1.,loss_new):
        raise ValueError(f"Conditional mismatch: {loss_new} vs {best['predicted_loss']}")
    if loss_new>=old_loss-1e-10:
        return factors,core,old_loss,0.,scan_time,None
    return best['factors'],core_new,loss_new,old_loss-loss_new,scan_time,best


def als_sweep(y, factors, indices):
    # Standard masked alternating least squares in mode-factor rows; reorthogonalize
    # and transport the gauge into the core after every mode update.
    factors=[u.copy() for u in factors]
    core,_=core_fit(y,factors,indices)
    for k,U in enumerate(factors):
        t,_=mode_coefficients(factors,core,indices,k)
        newU=U.copy()
        for i in range(U.shape[0]):
            mask=indices[:,k]==i
            if mask.sum()>=U.shape[1]:
                newU[i,:], *_ = np.linalg.lstsq(t[mask],y[mask],rcond=None)
        newU,Qr=np.linalg.qr(newU,mode='reduced')
        # Core transport: newU @ Qr = prior unsconstrained U update
        core=np.moveaxis(np.tensordot(Qr,core,axes=(1,k)),0,k)
        factors[k]=newU
        core,_=core_fit(y,factors,indices)
    core,loss=core_fit(y,factors,indices)
    return factors,core,loss


def synthetic(seed, shape, ranks, observed_fraction=.35, corruption_modes=1, noise=.005):
    rng=np.random.default_rng(seed)
    trueU=[orth(rng.normal(size=(n,r))) for n,r in zip(shape,ranks)]
    truecore=rng.normal(size=ranks)
    total=np.prod(shape)
    take=rng.permutation(total)
    m=int(total*observed_fraction)
    train_idx=np.array(np.unravel_index(take[:m],shape)).T
    test_idx=np.array(np.unravel_index(take[m:m+min(m,total-m)],shape)).T
    ytrain=predict(trueU,truecore,train_idx)+noise*rng.normal(size=m)
    ytest=predict(trueU,truecore,test_idx)
    factors=[v.copy() for v in trueU]
    # Replace last column in specified modes with a direction orthogonal to entire
    # true subspace. This simulates a genuinely incorrect early selection.
    for k in range(corruption_modes):
        v=rng.normal(size=shape[k]);v-=trueU[k]@(trueU[k].T@v);v/=np.linalg.norm(v)
        factors[k][:,-1]=v
    return ytrain,train_idx,ytest,test_idx,factors,trueU,truecore


def experiment(seed, shape, ranks, corruption_modes, steps=4, frac=.35, workers=1):
    y,idx,yt,it,baseU,_,_=synthetic(seed,shape,ranks,frac,corruption_modes)
    baseg,base_loss=core_fit(y,baseU,idx)
    results={'seed':seed,'shape':shape,'ranks':ranks,'corruption_modes':corruption_modes,
             'initial_train':base_loss,'initial_test': .5*float(np.sum((yt-predict(baseU,baseg,it))**2))}
    for method in ['continuous','finite','als']:
        f=[u.copy() for u in baseU]
        start=time.perf_counter()
        loss=base_loss
        gains=[];scan=[];chosen=[]
        for i in range(steps):
            if method=='als':
                f,g,loss=als_sweep(y,f,idx)
            else:
                f,g,loss,gain,dt,best=one_round(y,f,idx,method=method,workers=workers)
                gains.append(float(gain)); scan.append(float(dt));
                if best: chosen.append([int(best['mode']),int(best['column'])])
        g,loss=core_fit(y,f,idx)
        results[method]={'train':loss,'test':.5*float(np.sum((yt-predict(f,g,it))**2)),
            'seconds':time.perf_counter()-start,'gains':gains,'scan_seconds':scan,'chosen':chosen}
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=5)
    parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    cases=[((12,11,10,9),(2,2,2,2),1),
           ((8,8,7,7,6),(2,2,2,2,2),1),
           ((12,11,10,9),(2,2,2,2),2)]
    with threadpool_limits(limits=1):
        for shape,ranks,corruption_modes in cases:
            trials=[]
            for seed in range(args.seeds):
                r=experiment(seed,shape,ranks,corruption_modes,args.steps,workers=args.workers)
                trials.append(r)
                print(json.dumps({'case':str(shape)+'/'+str(corruption_modes),'seed':seed,
                  'initial':round(r['initial_test'],5),**{k:{'test':round(r[k]['test'],5), 'train':round(r[k]['train'],5), 'time':round(r[k]['seconds'],3),'chosen':r[k]['chosen']} for k in ['continuous','finite','als']}}),flush=True)
            for key in ['continuous','finite','als']:
                ratios=np.array([v[key]['test']/v['initial_test'] for v in trials])
                times=np.array([v[key]['seconds'] for v in trials])
                print('SUMMARY',shape,'wrong modes',corruption_modes,key,'median test ratio',round(float(np.median(ratios)),4),
                'median time s',round(float(np.median(times)),3),'mean test ratio',round(float(ratios.mean()),4),flush=True)

if __name__=='__main__':
    main()
