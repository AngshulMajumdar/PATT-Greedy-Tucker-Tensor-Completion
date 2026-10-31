"""Non-oracle enumerative h-way block rank exchange for the h-mode obstruction.
FULLY OBSERVED TOY ONLY. Algorithm sees target Y but never receives correct
receiver mode indices; enumerates all feasible receiver mode subsets.
Not the PATT benchmarked algorithm. Cost grows combinatorially with h.
"""
import time,itertools,csv
import numpy as np

def prod(T,M,k):return np.moveaxis(np.tensordot(M,T,axes=(1,k)),0,k)
def projection(Y,fac):
 T=Y.copy()
 for k,U in enumerate(fac):T=prod(T,U@U.T,k)
 return T
def subset_contraction(res,fac,recv):
 T=res
 for k,U in enumerate(fac):
  if k not in recv:T=prod(T,U.T,k)
 return T

def run(h,amplitude):
 d=2*h+2;n=3;eye=np.eye(3);shape=(n,)*d
 true_recv=tuple(range(h));extra=tuple(range(h,h+2));donors=tuple(range(h+2,d))
 fac=[eye[:,0:1].copy() for _ in range(d)]
 for k in donors:fac[k]=eye[:,[0,2]].copy()
 Y=np.zeros(shape);Y[(0,)*d]=1.
 ix=[0]*d
 for j in true_recv: ix[j]=1
 Y[tuple(ix)]=amplitude
 B=sum(n*U.shape[1] for U in fac)+int(np.prod([U.shape[1] for U in fac]))
 base_loss=.5*np.linalg.norm(Y-projection(Y,fac))**2
 t0=time.perf_counter();best=(float('inf'),None,None);ntry=0
 # Determine receiver subset by finite-size beam-free enumeration of all disjoint h-mode moves.
 for ds in itertools.combinations([k for k,U in enumerate(fac) if U.shape[1]>1],h):
  reduced=[U.copy() for U in fac]
  for k in ds:
   # full observed core: donor U^T projection, then remove smallest core slice
   reduced[k]=reduced[k][:,:-1]
  residual=Y-projection(Y,reduced)
  receivers=[k for k,U in enumerate(reduced) if k not in ds and U.shape[1]<n]
  for rs in itertools.combinations(receivers,h):
   candidate=[U.copy() for U in reduced]
   newr=[U.shape[1]+(k in rs) for k,U in enumerate(reduced)]
   cost=sum(n*r for r in newr)+int(np.prod(newr))
   if cost>B:continue
   ntry+=1
   H=subset_contraction(residual,reduced,rs)
   valid=True
   for k in rs:
    unfold=np.moveaxis(H,k,0).reshape(n,-1)
    unfold-=candidate[k]@(candidate[k].T@unfold)
    if np.linalg.norm(unfold)<1e-10:valid=False;break
    v=np.linalg.svd(unfold,full_matrices=False)[0][:,0]
    v-=candidate[k]@(candidate[k].T@v);v/=np.linalg.norm(v)
    candidate[k]=np.column_stack([candidate[k],v])
   if not valid:continue
   fitted=projection(Y,candidate)
   loss=.5*np.linalg.norm(Y-fitted)**2
   if loss<best[0]:best=(loss,ds,rs)
 return dict(h=h,d=d,amp=amplitude,n_states=ntry,init_loss=base_loss,
            final_loss=best[0],chosen_donors=str(best[1]),chosen_receivers=str(best[2]),
            recovered=int(best[0]<1e-20 and best[2]==true_recv),elapsed_s=time.perf_counter()-t0,
            budget=B)
if __name__=='__main__':
 rows=[]
 for h in [2,3,4]:
  for a in [0.3,0.6,0.9]:
   z=run(h,a);rows.append(z);print(z,flush=True)
 p=str(__import__('pathlib').Path(__file__).resolve().parent/'limited_block_lookahead_results.csv')
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
