"""Independent mechanism-level test of exact LS donor deletion and masked receiver SVD.
No labels/true factors enter construction after test instance generation; only used for evaluation.
"""
import argparse,csv,os,time
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from numpy.linalg import svd, norm
from scipy.linalg import lstsq

def orth(X):return np.linalg.qr(X,mode='reduced')[0]
def modeprod(T,M,k):return np.moveaxis(np.tensordot(M,T,axes=(1,k)),0,k)
def tucker(G,U):
 T=G
 for k,mat in enumerate(U):T=modeprod(T,mat,k)
 return T
def design(U,idx):
 arr=[u[idx[:,k],:] for k,u in enumerate(U)]
 v=arr[0]
 for a in arr[1:]:v=(v[:,:,None]*a[:,None,:]).reshape(len(idx),-1)
 return v
def fit(y,idx,U):
 A=design(U,idx);g=lstsq(A,y,lapack_driver='gelsy')[0].reshape(tuple(u.shape[1] for u in U));r=y-A@g.reshape(-1);return g,r,A

def generate(rng,shape=(8,9,7,7),noise=.001):
 Utrue=[orth(rng.normal(size=(n,2))) for n in shape]
 G=rng.normal(size=(2,2,2,2))
 Y=tucker(G,Utrue);Y=Y/np.sqrt(np.mean(Y**2));G=G/np.sqrt(np.mean(tucker(G,Utrue)**2))
 # Incorrect ranks: (3,1,2,2); donor 0 surplus and receiver 1 missing.
 v=rng.normal(size=shape[0]);v-=Utrue[0]@(Utrue[0].T@v);v/=norm(v)
 U=[np.column_stack([Utrue[0],v]),Utrue[1][:,:1].copy(),Utrue[2].copy(),Utrue[3].copy()]
 return Y,Utrue,U,v

def take_trial(seed,p,mask='mcar',noise_sd=.002):
 rng=np.random.default_rng(seed);Y,Ut,U,v0=generate(rng)
 N=Y.size;shape=Y.shape
 m=int(round(N*p));flat=np.arange(N)
 if mask=='block':
  block=np.zeros(shape,bool);block[int(shape[0]*.55):,int(shape[1]*.55):,:,:]=True
  eligible=flat[~block.reshape(-1)]; m=min(m,len(eligible));select=rng.choice(eligible,m,replace=False)
 else:select=rng.choice(N,m,replace=False)
 idx=np.array(np.unravel_index(select,shape)).T
 noise=rng.normal(scale=noise_sd,size=m)
 y=Y[tuple(idx.T)]+noise
 g_init,r_init,A_init=fit(y,idx,U)
 f0=.5*norm(r_init)**2
 # Ideal core and ideal donor zero vector (v0 is exactly the nullspace in chosen coordinates).
 # At sample level, SVD right-singular shape is donor x rest.
 Gn=np.moveaxis(g_init,0,0).reshape(3,-1)
 L,S,RT=svd(Gn,full_matrices=True)
 donor=L[:,-1]
 donor_sin=np.sqrt(max(0.,1.-(abs(donor[-1]))**2))
 Urot=[a.copy() for a in U]; Urot[0]=U[0]@L[:,:2]
 gminus,rminus,Aminus=fit(y,idx,Urot);fminus=.5*norm(rminus)**2
 ell=fminus-f0
 q=int(np.prod([Urot[k].shape[1] for k in range(4) if k!=1]))
 Kr=design([Urot[k] for k in range(4) if k!=1],idx[:,[0,2,3]])
 H=np.zeros((shape[1],q));np.add.at(H,idx[:,1],rminus[:,None]*Kr)
 V=Urot[1]
 Hp=H-V@(V.T@H)
 vh=svd(Hp,full_matrices=False)[0][:,0]
 vh-=V@(V.T@vh);vh/=max(norm(vh),1e-15)
 receiver_sin=np.sqrt(max(0.,1.-abs(vh@Ut[1][:,1])**2))
 def gain_of(v):
  if norm(v)<1e-10:return 0.,None
  vc=[x.copy() for x in Urot];vc[1]=np.column_stack([V,v])
  cg,res,AA=fit(y,idx,vc);return fminus-.5*norm(res)**2,AA
 Gact,Afinal=gain_of(vh);net=Gact-ell
 s=norm(Hp,ord=2)
 gramN=svd(Afinal,compute_uv=False)**2;lam_min=gramN[-1];lam_max=gramN[0]
 score=s*s/(2*max(lam_max,1e-15))-ell
 oldscore=.5*(s*s-S[-1]**2)
 # Receiver Gram residualized old features, directly mask aware (stronger valid bound).
 Zn=design([Urot[0],np.column_stack([V,vh]),Urot[2],Urot[3]],idx)
 # columns old are every other receiver index in core tensor flatten order, rj second.
 oldcols=np.arange(Zn.shape[1]).reshape((2,2,2,2))[:,0,:,:].reshape(-1)
 newcols=np.arange(Zn.shape[1]).reshape((2,2,2,2))[:,1,:,:].reshape(-1)
 Ao=Zn[:,oldcols];Znew=Zn[:,newcols]
 Qo=np.linalg.qr(Ao,mode='reduced')[0];W=Znew-Qo@(Qo.T@Znew)
 lw=svd(W,compute_uv=False)**2; lwmax=lw[0]
 cert=norm(Hp.T@vh)**2/(2*max(lwmax,1e-15))-ell
 # random-search reference for receiver optimality, not an exact optimizer
 best=Gact
 for _ in range(60):
  w=rng.normal(size=shape[1]);w-=V@(V.T@w);w/=norm(w)
  g,_=gain_of(w);best=max(best,g)
 fitted=fit(y,idx,[Urot[0],np.column_stack([V,vh]),Urot[2],Urot[3]])[0]
 recon=tucker(fitted,[Urot[0],np.column_stack([V,vh]),Urot[2],Urot[3]])
 nmse=norm(Y-recon)**2/norm(Y)**2
 # Subspace-specific Gram only; this is not the adaptive concentration theorem.
 dims=(2,2,2,2);eigs=(gramN/p);ratio=float(eigs[0]/max(eigs[-1],1e-15))
 return dict(seed=seed,p=p,mask=mask,nobs=m,donor_sin=donor_sin,receiver_sin=receiver_sin,
             sigma_min=float(S[-1]),gap=float(S[-2]),donor_loss=float(ell),
             s=float(s),gross_receiver_gain=float(Gact),net_gain=float(net),
             original_cert=float(oldscore),mask_cert=float(cert),weak_gram_cert=float(score),
             gain_ratio_to_60_random=float(Gact/best) if best>0 else 1.,
             gram_min_scaled=float(eigs[-1]),gram_max_scaled=float(eigs[0]),gram_cond=ratio,
             nmse=float(nmse),success=int(nmse<=1e-3),oldcert_pass=int(oldscore>0),
             newcert_pass=int(cert>0),training_improved=int(net>1e-10),cert_sound=int(net+1e-7>=cert))

def obstruction():
  e=np.eye(3)
  Y=np.einsum('i,j,k,l->ijkl',e[:,0],e[:,0],e[:,0],e[:,0])+np.einsum('i,j,k,l->ijkl',e[:,1],e[:,1],e[:,0],e[:,0])
  Uf=[e[:,[0]],e[:,[0]],e[:,[0,2]],e[:,[0,2]]]
  idx=np.indices(Y.shape).reshape(4,-1).T
  g,r,_=fit(Y.reshape(-1),idx,Uf)
  assert np.isclose(norm(r)**2,1)
  gains=[]
  for donor in [2,3]:
   # delete redundant, increase receiver 0 or 1, still no gain
   upd=[a.copy() for a in Uf];upd[donor]=e[:,[0]]
   for recv in [0,1]:
    pp=[a.copy() for a in upd];pp[recv]=e[:,[0,1]]
    gc,rc,_=fit(Y.reshape(-1),idx,pp)
    gains.append((donor,recv,float(norm(r)**2-norm(rc)**2)))
  # joint block transfers: donors (2,3) -> receivers (0,1)
  UB=[e[:,[0,1]],e[:,[0,1]],e[:,[0]],e[:,[0]]]
  gc,rc,_=fit(Y.reshape(-1),idx,UB)
  assert norm(rc)<1e-12
  return gains,float(norm(rc))

if __name__=='__main__':
 pa=argparse.ArgumentParser();pa.add_argument('--nseeds',type=int,default=20);pa.add_argument('--output',required=True);args=pa.parse_args()
 rows=[];t=time.time()
 for mask in ('mcar','block'):
  for p in (.10,.25,.40):
   for seed in range(args.nseeds):
    rows.append(take_trial(seed+2026,p,mask))
   rs=[r for r in rows if r['mask']==mask and r['p']==p]
   print(mask,p,'success',sum(r['success'] for r in rs),'new cert positive',sum(r['newcert_pass'] for r in rs),'median-angle',np.median([r['donor_sin'] for r in rs]),'median-cond',np.median([r['gram_cond'] for r in rs]),'t',time.time()-t,flush=True)
 with open(args.output,'w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
 print('obstruction',obstruction())
