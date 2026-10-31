"""Numerical sanity checks for the paper's donor/receiver certificate.
Not a substitute for the mathematical proof. Uses exact sampled least squares.
"""
import itertools
import numpy as np

def design(U, inds):
    a=U[0][inds[:,0],:]
    for k in range(1,len(U)):
        a=np.einsum('mi,mj->mij',a,U[k][inds[:,k],:]).reshape(len(inds),-1)
    return a

def core_rotate(g, L, k):
    g=np.tensordot(L.T,g,axes=(1,k))
    return np.moveaxis(g,0,k)

def check_one(seed):
    rg=np.random.default_rng(seed)
    shp=(5,6,5); rr=(3,2,2);d=3
    fac=[np.linalg.qr(rg.standard_normal((n,r)))[0] for n,r in zip(shp,rr)]
    indices=np.array(list(itertools.product(*(range(n) for n in shp))),dtype=int)
    inds=indices[rg.random(len(indices)) < rg.choice([0.2,0.4,0.7,1.0])]
    if len(inds)<np.prod(rr)+5:return None
    y=rg.standard_normal(len(inds))
    A0=design(fac,inds)
    gvec=np.linalg.lstsq(A0,y,rcond=None)[0]
    F0=np.linalg.norm(y-A0@gvec)**2/2
    g=gvec.reshape(rr)
    i,j=0,1
    unfold=np.moveaxis(g,i,0).reshape(rr[i],-1)
    L,s,_=np.linalg.svd(unfold,full_matrices=False)
    si=s[-1]
    U=[x.copy() for x in fac]
    U[i]=U[i]@L[:,:-1]
    A=design(U,inds)
    alpha=np.linalg.lstsq(A,y,rcond=None)[0]
    Fpr=np.linalg.norm(y-A@alpha)**2/2
    e=y-A@alpha
    q=int(np.prod([len(uk[0]) for k,uk in enumerate(U) if k!=j]))
    h=np.zeros((shp[j],q))
    rows=np.ones((len(inds),1))
    for k in range(d):
        if k!=j:rows=np.einsum('ma,mb->mab',rows,U[k][inds[:,k],:]).reshape(len(inds),-1)
    np.add.at(h, inds[:,j],e[:,None]*rows)
    h-=U[j]@(U[j].T@h)
    v=np.linalg.svd(h,full_matrices=False)[0][:,0]
    sv=np.linalg.svd(h,compute_uv=False)[0]
    U[j]=np.column_stack([U[j],v])
    Aplus=design(U,inds)
    b=np.linalg.lstsq(Aplus,y,rcond=None)[0]
    Fplus=np.linalg.norm(y-Aplus@b)**2/2
    delta=F0-Fplus
    cert=(sv*sv-si*si)/2
    # numerical ratio check for the donor part
    assert Fpr-F0 <= 0.5*si*si+1e-8,(seed,'donor',Fpr-F0,si*si/2)
    assert Fpr-Fplus >=0.5*sv*sv-1e-8,(seed,'receiver',Fpr-Fplus,sv*sv/2)
    assert delta>=cert-1e-8,(seed,'certificate',delta,cert)
    return delta,cert,len(inds)

def rank_graph():
    from collections import deque
    n=(4,5,6);b=94;start=(3,2,6);goal=(4,5,2)
    cost=lambda r:sum(ni*ri for ni,ri in zip(n,r))+np.prod(r)
    queue=deque([start]);seen={start}
    while queue:
        r=queue.popleft()
        for i,j in itertools.permutations(range(3),2):
            z=list(r);z[i]-=1;z[j]+=1;z=tuple(z)
            if min(z)>=1 and all(x<=ni for x,ni in zip(z,n)) and cost(z)<=b and z not in seen:
                seen.add(z);queue.append(z)
    assert seen=={(3,2,6),(4,1,6)},seen
    assert goal not in seen
    return seen

if __name__=='__main__':
    vals=[check_one(s) for s in range(200)]
    vals=[v for v in vals if v is not None]
    print(f'Certificate verification passed in {len(vals)} randomized sampled trials.')
    print('minimum observed margin (true gain minus certificate):',min(d-c for d,c,_ in vals))
    print('reachable rank vectors:',rank_graph())

def check_exact_one_swap(seed):
    rg=np.random.default_rng(seed+10000)
    shp=(7,8,6); rtrue=(2,3,2); i,j=0,1
    Ut=[np.linalg.qr(rg.standard_normal((n,r)))[0] for n,r in zip(shp,rtrue)]
    z=rg.standard_normal((shp[0],1));z-=Ut[0]@(Ut[0].T@z);z/=np.linalg.norm(z)
    U=[np.column_stack([Ut[0],z]), Ut[1][:,:-1], Ut[2].copy()]
    idx=np.array(list(itertools.product(*(range(n) for n in shp))),dtype=int)
    y=design(Ut,idx)@rg.standard_normal(np.prod(rtrue))
    A0=design(U,idx)
    gvec=np.linalg.lstsq(A0,y,rcond=None)[0]
    g=gvec.reshape([len(uk[0]) for uk in U])
    L,s,_=np.linalg.svd(np.moveaxis(g,i,0).reshape(3,-1),full_matrices=False)
    # Exact recovery requires a unique null donor direction.
    assert s[-1]<1e-10 and s[-2]>1e-6,(seed,s)
    U[i]=U[i]@L[:,:-1]
    A=design(U,idx)
    alpha=np.linalg.lstsq(A,y,rcond=None)[0]
    e=y-A@alpha
    rows=np.ones((len(idx),1))
    for k in range(3):
        if k!=j:rows=np.einsum('ma,mb->mab',rows,U[k][idx[:,k],:]).reshape(len(idx),-1)
    h=np.zeros((shp[j],rows.shape[1]))
    np.add.at(h,idx[:,j],e[:,None]*rows)
    h-=U[j]@(U[j].T@h)
    recovered=np.linalg.svd(h,full_matrices=False)[0][:,0]
    assert abs(recovered@Ut[j][:,-1]) > 1-1e-10
    U[j]=np.column_stack([U[j],recovered])
    est=design(U,idx)@np.linalg.lstsq(design(U,idx),y,rcond=None)[0]
    err=np.linalg.norm(est-y)/np.linalg.norm(y)
    assert err<1e-10,(seed,err)
    return err

if __name__=='__main__':
    errs=[check_exact_one_swap(i) for i in range(40)]
    print(f'Exact one-swap verification passed in {len(errs)} full-observation cases; largest relative error {max(errs):.2e}.')
