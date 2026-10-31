"""Actual PATT post-donor-refit receiver: certified training gain BEFORE expanded core solve.
72 fresh trials. All candidates/bench outcomes computed from observed values only;
underlying tensor used only to generate data, never to select candidate."""
import os
for t in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[t]='1'
import sys,csv
import numpy as np
from numpy.linalg import norm,svd
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
from mechanism_experiment import generate,fit,design,modeprod

def trial(seed,p,mask):
 rng=np.random.default_rng(seed);Y,Ut,U,_=generate(rng)
 shape=Y.shape;N=Y.size;m=round(N*p)
 eligible=np.arange(N)
 if mask=='block':
  allowed=np.ones(shape,bool);allowed[int(shape[0]*.55):,int(shape[1]*.55):,:,:]=False
  eligible=eligible[allowed.ravel()]
 sel=rng.choice(eligible,min(m,len(eligible)),replace=False)
 idx=np.array(np.unravel_index(sel,shape)).T
 y=Y.reshape(-1)[sel]+rng.normal(scale=.002,size=len(sel))
 G0,r0,A0=fit(y,idx,U);f0=.5*float(r0@r0)
 # Exact gauge rotation followed by deletion, no additional fitting yet
 L,_,_=svd(G0.reshape(3,-1),full_matrices=False)
 Urot=[a.copy() for a in U];Urot[0]=Urot[0]@L
 Grot=modeprod(G0,L.T,0)
 pruned=[x.copy() for x in Urot];pruned[0]=Urot[0][:,:2]
 predold=design(Urot,idx)@Grot.ravel()
 predraw=design(pruned,idx)@Grot[:2].ravel()
 deleted=predold-predraw
 raw=y-predraw
 Gr,eref,Ar=fit(y,idx,pruned); fminus=.5*float(eref@eref)
 ell=fminus-f0
 q=int(np.prod([pruned[k].shape[1] for k in [0,2,3]]))
 K=design([pruned[k] for k in [0,2,3]],idx[:,[0,2,3]])
 n=shape[1];V=pruned[1]
 def H_of(res):
  M=np.zeros((n,q)); np.add.at(M,idx[:,1],res[:,None]*K)
  return M - V@(V.T@M)
 Hraw=H_of(raw);Href=H_of(eref)
 v=svd(Href,full_matrices=False)[0][:,0]
 v-=V@(V.T@v);v/=norm(v)
 sraw=float(norm(Hraw,2)); sr=float(norm(Href,2))
 epsH=float(norm(Hraw-Href,2)); projection=float(norm(raw-eref))
 bhraw=float(norm(Hraw.T@v));bhref=float(norm(Href.T@v))
 vU=[x.copy() for x in pruned];vU[1]=np.column_stack([V,v])
 gf,rf,Af=fit(y,idx,vU); gain=f0-.5*float(rf@rf)
 cert_post=0.5*bhref**2 -ell
 cert_post_lam=0.5*bhref**2 -ell # can tighten via Gram upper bound, not needed
 cert_raw=0.5*bhraw**2-0.5*float(deleted@deleted)
 cert_bridge=0.5*max(0,sraw-2*epsH)**2-0.5*float(deleted@deleted)
 return dict(seed=seed,p=p,mask=mask,donor_loss=ell,
    full_gain=gain,post_prune_cert=cert_post,raw_direction_cert=cert_raw,
    bridge_cert=cert_bridge,score_refit=sr,
    score_raw=sraw,raw_delta_contraction=epsH,
    projected_deleted_norm=projection,deleted_norm=float(norm(deleted)),
    bridge_score_valid=int(bhraw+1e-7>=sraw-2*epsH),
    post_bound_valid=int(cert_post<=gain+1e-7),
    raw_bound_valid=int(cert_raw<=gain+1e-7),
    bridge_bound_valid=int(cert_bridge<=gain+1e-7),
    positive_post=int(cert_post>1e-8),positive_bridge=int(cert_bridge>1e-8),
    exchange_improved=int(gain>1e-8))
if __name__=='__main__':
 out=[trial(9500+i,p,mask) for mask in ['mcar','block'] for p in [.1,.25,.4] for i in range(12)]
 with open(str(__import__('pathlib').Path(__file__).resolve().parent/'receiver_bridge_diagnostics.csv'),'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
 for mask in ['mcar','block']:
  for p in [.1,.25,.4]:
   r=[x for x in out if x['mask']==mask and x['p']==p]
   print(mask,p, 'post+',sum(x['positive_post'] for x in r),'bridge+',sum(x['positive_bridge'] for x in r), 'improved',sum(x['exchange_improved'] for x in r),'valid post',sum(x['post_bound_valid'] for x in r),'valid bridge',sum(x['bridge_bound_valid'] for x in r))
