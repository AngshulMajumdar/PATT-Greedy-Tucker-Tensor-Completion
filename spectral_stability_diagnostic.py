"""Independent numerical diagnostic: rank-one ideal residual and off-signal perturbation bound.
This is an oracle-only test, not a prediction-time certificate. No oracle info enters
mode factor estimation or reconstruction."""
import os
for t in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[t]='1'
import sys,csv
import numpy as np
from numpy.linalg import norm,svd
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
from mechanism_experiment import generate,fit,design,modeprod,tucker

def test(seed,p,mask):
    rng=np.random.default_rng(seed);Y,Ut,U,_=generate(rng)
    shp=Y.shape; N=Y.size; m=round(p*N)
    pool=np.arange(N)
    if mask=='block':
        keep=np.ones(shp,bool);keep[int(shp[0]*.55):,int(shp[1]*.55):,:,:]=False
        pool=pool[keep.ravel()]
    flat=rng.choice(pool,min(m,len(pool)),replace=False)
    idx=np.array(np.unravel_index(flat,shp)).T
    y=Y.ravel()[flat]+rng.normal(scale=.002,size=len(flat));p_eff=len(flat)/N
    G0,r0,A0=fit(y,idx,U)
    left,sv,_=svd(G0.reshape(3,-1),full_matrices=False)
    Um=[a.copy() for a in U]; Um[0]=U[0]@left[:,:2]
    gm,rm,A=fit(y,idx,Um)
    K=design([Um[0],Um[2],Um[3]],idx[:,[0,2,3]])
    H=np.zeros((shp[1],K.shape[1]));np.add.at(H,idx[:,1],rm[:,None]*K);H=H/p_eff
    V=Um[1];P=V@V.T;H=(np.eye(shp[1])-P)@H
    # Full tensor projected residual for these *same estimated* nonreceiver factors.
    F=Y.copy()
    for k,uk in enumerate(Um):F=modeprod(F,uk@uk.T,k)
    Ef=Y-F
    Kfull=np.kron(np.kron(Um[0],Um[2]),Um[3])
    H0=(np.eye(shp[1])-P)@(np.moveaxis(Ef,1,0).reshape(shp[1],-1)@Kfull)
    u,ss,_=svd(H0,full_matrices=False);u=u[:,0];s=ss[0]
    # Remainder of H0 is numerically rank-one in constructed conditions.
    E=H-H0;eperp=E-np.outer(u,u@E)
    a=u@H
    A_sig=float(a@a);D=float(norm(eperp,2)**2);b=float(norm(eperp@a))
    margin=A_sig-D
    bound=np.arctan2(b,max(margin,0)) if margin>0 else np.pi/2
    v=svd(H,full_matrices=False)[0][:,0]
    theta=np.arccos(np.clip(abs(u@v),-1,1));
    # Add legacy bulk Wedin condition for comparison.
    zeta=float(norm(E,2));old=s>zeta
    return dict(seed=seed,mask=mask,p=p,ideal_rank2=float(ss[1] if len(ss)>1 else 0),
       ideal_signal=s,bulk_error=zeta,old_wedin_eligible=int(old),
       offsignal_error=float(norm(eperp,2)),targeted_margin=margin,
       targeted_eligible=int(margin>0),observed_angle_deg=float(np.degrees(theta)),
       targeted_angle_bound_deg=float(np.degrees(bound)),
       bound_valid=int(theta<=bound+1e-7))

if __name__=='__main__':
    out=[]
    for mask in ['mcar','block']:
      for p in [.1,.25,.4]:
       for i in range(12): out.append(test(7600+i,p,mask))
    with open(str(__import__('pathlib').Path(__file__).resolve().parent/'spectral_stability_diagnostics.csv'),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
    for mask in ['mcar','block']:
      for p in [.1,.25,.4]:
        r=[x for x in out if x['mask']==mask and x['p']==p];print(mask,p,'old',sum(x['old_wedin_eligible'] for x in r), 'new',sum(x['targeted_eligible'] for x in r),'bound valid',sum(x['bound_valid'] for x in r),'median-angle',np.median([x['observed_angle_deg'] for x in r]),'max ideal rank2',max(x['ideal_rank2'] for x in r))
