"""Image-only ARE/TMac rank-scanning Python adaptation.

Based on official MATLAB image code:
 https://github.com/lymch/ARE/blob/main/Tucker/Tucker_ARE.m

WARNING: NOT an exact faithful translation, because released MATLAB uses
  * unbound `addrank_Tucker` in Tucker_ARE.m (addrank.m is different);
  * hard-coded 256x256x3 to 256x256x12 replication;
  * RS4Tucker.m places the full input image X (incl. missing entries) in
    the first 3 layers. The demo calls Tucker_ARE(X1,X1,known) with X1
    being the complete test image: this would leak the ground truth into
    the reconstruction procedure if translated literally.

We therefore implement a TRAIN-ONLY repaired variant. It keeps the intent:
linear interpolation initialization, SSIM-scored local unfolding rank updates,
repeated TMac solves, pseudo-observation refinement; fixes variable dimensions,
missing definition and full-image leakage. The output is NOT official ARE.
Only applies to HxWx3 images, and should not be compared on general-order tensors.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import time
import numpy as np
from scipy.interpolate import griddata
from skimage.metrics import structural_similarity as ssim
from tmac_matlab_python_port import tmac,evaluate


def interpolate_observed(image,mask):
    if image.ndim!=3 or image.shape[-1]!=3:raise ValueError('ARE only supports 3-channel images')
    h,w,c=image.shape;result=np.empty_like(image,dtype=float)
    rows,cols=np.indices((h,w))
    for k in range(3):
        rr,cc=np.where(mask[:,:,k])
        if not len(rr):raise ValueError('No observed entries in channel')
        vals=image[rr,cc,k]
        if len(vals)>=3:
            try:
                rec=griddata(np.column_stack([rr,cc]),vals,(rows,cols),method='linear',fill_value=np.nan)
            except Exception:rec=np.full((h,w),np.nan)
        else:rec=np.full((h,w),np.nan)
        holes=~np.isfinite(rec)
        if holes.any():rec[holes]=griddata(np.column_stack([rr,cc]),vals,(rows[holes],cols[holes]),method='nearest')
        rec[mask[:,:,k]]=image[:,:,k][mask[:,:,k]]
        result[:,:,k]=rec
    return result

def observed_tuple(image,mask):
    idx=np.column_stack(np.nonzero(mask))
    return image[tuple(idx.T)],idx

def score_ssim(ref,img):
    dr=max(float(np.max(ref)-np.min(ref)),1e-5)
    return float(ssim(ref,img,channel_axis=2,data_range=dr,
                      win_size=max(3,min(7,2*(min(ref.shape[:2])//2)-1))))

def are_image(observed,mask,*,seed=0,rounds=2,rank_init=(1,1,1),
              rank_bounds=((1,7),(1,7),(1,3)),offsets=(-2,-1,0,1,2),
              maxit_tmac=20,augment_fraction=.5):
    if observed.shape!=mask.shape or observed.ndim!=3 or observed.shape[2]!=3:
        raise ValueError('Three-channel observed image and same-shape mask required')
    # The image argument may contain anything at missing pixels: never use them.
    shape=observed.shape
    clean_obs=np.where(mask,observed,0.0)
    proxy=interpolate_observed(clean_obs,mask)
    rank=list(rank_init);hist=[];start=time.perf_counter()
    train=observed_tuple(clean_obs,mask)
    fullidx=np.array(np.unravel_index(np.arange(np.prod(shape)),shape)).T
    current=proxy.copy()
    rng=np.random.default_rng(55000+seed)
    for step in range(rounds):
        for mode in range(3):
            pool=sorted(set(min(rank_bounds[mode][1],max(rank_bounds[mode][0],rank[mode]+int(s))) for s in offsets))
            best_score=-np.inf;best_rank=rank[mode];best=current
            for r in pool:
                cand_r=list(rank);cand_r[mode]=int(r)
                # Pseudo-observation refinement; every pseudo-value is inferred
                # exclusively from observed training entries, never hidden truth.
                pred_img=np.where(mask,clean_obs,current)
                missing=np.column_stack(np.where(~mask))
                pseudo_count=int(len(missing)*augment_fraction)
                if pseudo_count>0:
                    pick=missing[rng.choice(len(missing),pseudo_count,replace=False)]
                    index=np.vstack([train[1],pick])
                    vals=np.concatenate([train[0],pred_img[tuple(pick.T)]])
                else:index=train[1];vals=train[0]
                model_input=(vals,index)
                # Validation is not ground truth: score against the interpolated
                # image, as the official wrapper does with Xinter.
                proxy_data=(proxy[tuple(fullidx.T)],fullidx)
                run,factors,weights=tmac(model_input,proxy_data,shape,cand_r,
                    seed=seed+1000*step+100*mode+int(r),rank_policy='fixed',maxit=maxit_tmac,
                    stop_val_nmse=-1,return_state=True)
                result=evaluate(factors,weights,shape,fullidx).reshape(shape)
                result[mask]=clean_obs[mask]
                s=score_ssim(proxy,result)
                if s>best_score:
                    best_score=s;best_rank=r;best=result
            rank[mode]=best_rank
            current=best
            hist.append({'round':step,'mode':mode,'rank':rank.copy(),
                         'ssim_to_interpolation':best_score,
                         'time_s':time.perf_counter()-start})
    current[mask]=clean_obs[mask]
    return {'reconstruction':current,'ranks':rank,'history':hist,
            'total_s':time.perf_counter()-start,
            'provenance':'train-only repaired image-specific Python adaptation, NOT exact ARE MATLAB'}

if __name__=='__main__':
 from skimage import data,transform,img_as_float
 img=transform.resize(img_as_float(data.astronaut()),(16,16),anti_aliasing=True)
 rng=np.random.default_rng(42);mask=rng.random(img.shape)<.4
 out=are_image(np.where(mask,img,0),mask,rounds=1,rank_init=(2,2,2),
               rank_bounds=((1,4),(1,4),(1,3)),offsets=(-1,0,1),maxit_tmac=10)
 print('ARE image adaptation:',out['ranks'],out['total_s'],
       'heldout NMSE',np.mean((out['reconstruction'][~mask]-img[~mask])**2)/np.mean(img[~mask]**2))
