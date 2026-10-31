"""Ground-truth-audited local refinement after an actual sampled rank exchange.
True factors appear only in principal-angle and held-out NMSE calculations.
This is *not* a numerical proof of the full-sweep-map Jacobian contraction.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,csv
import numpy as np
from pathlib import Path
from numpy.linalg import norm,svd
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mechanism_experiment import generate,fit,design
sys.path.insert(0,str(Path(__file__).resolve().parent))
from patt_tucker_correction_experiment import als_sweep,core_fit,predict

def one(seed,pp,mask):
    rng=np.random.default_rng(seed);Y,Ut,U,_=generate(rng)
    N=Y.size;shp=Y.shape
    pool=np.arange(N)
    if mask=='block':
        keep=np.ones(shp,bool);keep[int(shp[0]*.55):,int(shp[1]*.55):,:,:]=False
        pool=pool[keep.ravel()]
    ids=rng.choice(pool,min(round(N*pp),len(pool)),replace=False)
    idx=np.array(np.unravel_index(ids,shp)).T
    y=Y.ravel()[ids]+rng.normal(scale=.002,size=len(ids))
    g0,r0,_=fit(y,idx,U)
    sv=svd(g0.reshape(3,-1),full_matrices=False)[0]
    f=[u.copy() for u in U];f[0]=U[0]@sv[:,:2]
    gminus,rm,_=fit(y,idx,f)
    ph=design([f[0],f[2],f[3]],idx[:,[0,2,3]])
    B=np.zeros((shp[1],ph.shape[1]));np.add.at(B,idx[:,1],rm[:,None]*ph)
    B-=f[1]@(f[1].T@B)
    v=svd(B,full_matrices=False)[0][:,0];v-=f[1]@(f[1].T@v);v/=norm(v)
    f[1]=np.column_stack([f[1],v]);g,loss=core_fit(y,f,idx)
    allidx=np.array(np.unravel_index(np.setdiff1d(np.arange(N),ids)[:],shp)).T
    # test on entire disjoint holdout, never used during fitting
    yt=Y[tuple(allidx.T)]
    ranks=[u.shape[1] for u in f]
    def angle(U1,U2):
        return float(norm(U2-U1@(U1.T@U2),2))
    ini_angles=[angle(f[k],Ut[k]) for k in range(4)]
    nmse0=norm(predict(f,g,allidx)-yt)**2/max(norm(yt)**2,1e-30)
    hist=[nmse0]
    for sw in range(10):
        f,g,loss=als_sweep(y,f,idx)
        hist.append(norm(predict(f,g,allidx)-yt)**2/max(norm(yt)**2,1e-30))
    finalangles=[angle(f[k],Ut[k]) for k in range(4)]
    return dict(seed=seed,p=pp,mask=mask,initial_max_principal_sin=max(ini_angles),
        final_max_principal_sin=max(finalangles),initial_test_nmse=nmse0,
        test_nmse_after3=hist[3],test_nmse_after10=hist[10],
        reached_target=int(hist[10]<=.001),improvement_factor=nmse0/max(hist[10],1e-30),
        history=';'.join(f'{x:.9g}' for x in hist))

if __name__=='__main__':
 out=[]
 with threadpool_limits(limits=1):
    for mask in ['mcar','block']:
        for p in [.02,.1,.25]:
            for seed in range(9300,9306):
                a=one(seed,p,mask);out.append(a)
            z=[a for a in out if a['mask']==mask and a['p']==p]
            print(mask,p,'success10',sum(a['reached_target'] for a in z), 'medianangle',np.median([a['initial_max_principal_sin'] for a in z]),flush=True)
 with Path(__file__).with_name('post_exchange_basin_results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
