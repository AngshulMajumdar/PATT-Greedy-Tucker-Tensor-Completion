"""Oracle-only cross-check of targeted off-signal progress theorem.
Nonreceiver Tucker factors and receiver chosen by actual post-prune SVD on masked data;
true receiver u is used only in computing the theoretical A_u. Training improvement
and sample Gram are evaluated on observed entries. No ground-truth leakage.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,csv,sys
from numpy.linalg import norm,svd
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mechanism_experiment import generate,fit,design,modeprod
from threadpoolctl import threadpool_limits

def one(seed,p,mask):
    rng=np.random.default_rng(seed);Y,Ut,U,_=generate(rng)
    N=Y.size;shape=Y.shape;m=round(p*N)
    pool=np.arange(N)
    if mask=='block':
        keep=np.ones(shape,bool);keep[int(shape[0]*.55):,int(shape[1]*.55):,:,:]=False
        pool=pool[keep.ravel()]
    flat=rng.choice(pool,min(m,len(pool)),replace=False)
    idx=np.array(np.unravel_index(flat,shape)).T
    y=Y.ravel()[flat]+rng.normal(scale=.002,size=len(flat)); pp=len(flat)/N
    g0,r0,A0=fit(y,idx,U)
    L,_,_=svd(g0.reshape(3,-1),full_matrices=False)
    Um=[a.copy() for a in U]; Um[0]=U[0]@L[:,:2]
    gm,rm,Aminus=fit(y,idx,Um)
    ell=.5*(rm@rm-r0@r0)
    Kr=design([Um[0],Um[2],Um[3]],idx[:,[0,2,3]])
    H=np.zeros((shape[1],Kr.shape[1]));np.add.at(H,idx[:,1],rm[:,None]*Kr)
    V=Um[1]; H=(np.eye(shape[1])-V@V.T)@H
    hnorm=H/pp
    # Oracle receiver u: full-data projected residual of exactly same nonreceiver spaces.
    F=Y.copy()
    for k,uk in enumerate(Um): F=modeprod(F,uk@uk.T,k)
    Ef=Y-F
    Kfull=np.kron(np.kron(Um[0],Um[2]),Um[3])
    H0=(np.eye(shape[1])-V@V.T)@(np.moveaxis(Ef,1,0).reshape(shape[1],-1)@Kfull)
    u,_,_=svd(H0,full_matrices=False);u=u[:,0]
    v,_,_=svd(H,full_matrices=False);v=v[:,0];v-=V@(V.T@v);v/=norm(v)
    # projected new sampled feature Gram: no oracle info
    Z=v[idx[:,1],None]*Kr
    W=Z-Aminus@np.linalg.lstsq(Aminus,Z,rcond=None)[0]
    Lambda=norm(W,2)**2
    # Actual expanded core refit
    Up=[u0.copy() for u0 in Um]; Up[1]=np.column_stack([V,v])
    _,re,_=fit(y,idx,Up)
    delta=.5*(r0@r0-re@re)
    Au=float(norm(hnorm.T@u)**2)
    off=hnorm-np.outer(u,u@hnorm)
    Du=float(norm(off,2)**2)
    Bu=float(norm(off@(hnorm.T@u)))
    margin=Au-Du
    tanmax=Bu/margin if margin>0 else np.inf
    LBa=(pp**2*Au)/(2*Lambda)-ell if Lambda>0 else -ell
    # angle-dependent lower bound, most useful for *approximate* receiver eigensolvers
    anglelb=(np.sqrt(Au)-np.sqrt(Du)*tanmax)/np.sqrt(1+tanmax*tanmax) if margin>0 else -np.inf
    LBangle=(pp**2*max(anglelb,0)**2)/(2*Lambda)-ell if Lambda>0 and margin>0 else float('-inf')
    return dict(seed=seed,mask=mask,p=p,applicable=int(margin>0),
        A_u=Au,D_u=Du,B_u=Bu,targeted_margin=margin,
        actual_donor_loss=ell,gram_lambda_max=Lambda,
        actual_net_training_gain=delta,targeted_gain_lower_bound=LBa,
        angle_gain_lower_bound=LBangle,
        angle_gain_positive=int(LBangle>0),targeted_gain_positive=int(LBa>0),
        certificate_valid=int(LBa<=delta+1e-6))
if __name__=='__main__':
    out=[]
    with threadpool_limits(limits=1):
        for mask in ['mcar','block']:
            for p in [.1,.25,.4]:
                for i in range(12):
                    seed=7600+i
                    d=one(seed,p,mask);out.append(d)
                z=[a for a in out if a['mask']==mask and a['p']==p]
                print(mask,p,'eligible',sum(a['applicable'] for a in z), 'cert+',sum(a['targeted_gain_positive'] for a in z), 'valid',sum(a['certificate_valid'] for a in z),flush=True)
    with Path(__file__).with_name('targeted_progress_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
