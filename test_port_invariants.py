"""Finite-difference and no-data-leakage invariant checks for Python ports."""
import numpy as np
from threadpoolctl import threadpool_limits
from tram_matlab_python_port import orth_init,core_from_sampled,predict,tangent_gradient,tangent_observed,tucker_retract
from are_image_python_port import are_image


def test_tram_directional_derivative():
 rng=np.random.default_rng(9)
 shape=(9,8,7);r=(2,2,2)
 x=orth_init(shape,r,1)
 idx=np.column_stack([rng.integers(0,n,80) for n in shape]);y=rng.normal(size=len(idx))
 x.core=core_from_sampled(y,idx,x.factors)
 e=predict(x,idx)-y
 dc,du=tangent_gradient(x,idx,e)
 deriv=tangent_observed(x,idx,dc,du)
 eps=1e-6
 finite=(predict(tucker_retract(x,dc,du,eps),idx)-predict(x,idx))/eps
 relative=float(np.linalg.norm(finite-deriv)/max(np.linalg.norm(deriv),1e-12))
 assert relative<1e-3,relative
 assert all(np.linalg.norm(f.T@f-np.eye(f.shape[1]))<1e-8 for f in x.factors)
 return relative


def test_are_no_hidden_truth_dependency():
 rng=np.random.default_rng(88)
 im=rng.random((8,8,3));mask=rng.random(im.shape)<.55
 masked=np.where(mask,im,0.)
 corrupted=masked.copy();corrupted[~mask]=1000*rng.normal(size=np.sum(~mask))
 kws=dict(seed=10,rounds=1,rank_init=(2,2,2),rank_bounds=((1,3),(1,3),(1,3)),offsets=(0,),maxit_tmac=5)
 a=are_image(masked,mask,**kws)['reconstruction']
 b=are_image(corrupted,mask,**kws)['reconstruction']
 assert np.array_equal(a,b),'Hidden values leaked into ARE image reconstruction'
 return np.max(abs(a-b))

if __name__=='__main__':
 with threadpool_limits(limits=1):
  print('TRAM retraction derivative error:',test_tram_directional_derivative())
  print('ARE hidden-value invariance:',test_are_no_hidden_truth_dependency())
