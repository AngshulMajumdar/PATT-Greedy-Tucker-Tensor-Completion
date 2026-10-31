"""Re-create GitHub README charts from the archived CSV files (no simulated data)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':11,'axes.titlesize':15,'axes.labelsize':11,'figure.dpi':160,
                     'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
COL={'PATT-rank-exchange':'#1966ae','Tucker-ALS':'#9b9fa4','Tucker-ALS-24':'#606a75',
     'Tucker-HOSVD+ALS':'#e59a42','Tucker-rank-search-8':'#8c64b7'}
SHORT={'PATT-rank-exchange':'PATT: rank exchange','Tucker-ALS':'Tucker ALS',
       'Tucker-ALS-24':'ALS (24 sweeps)','Tucker-HOSVD+ALS':'HOSVD + ALS',
       'Tucker-rank-search-8':'8-candidate rank search'}
summary=pd.read_csv(ROOT/'results_threshold_summary.csv')
raw=pd.read_csv(ROOT/'results_primary_25pct.csv')

def savefig(name):
    plt.savefig(ROOT/name,bbox_inches='tight',facecolor='white');plt.close()

def grouped_success(setup,methods,filename,title):
    fig,ax=plt.subplots(figsize=(10,4.9));xx=np.arange(3,7);w=.78/len(methods)
    for j,m in enumerate(methods):
        vals=[int(summary[(summary.rank_setup==setup)&(summary.order==o)&(summary.method==m)].successes.iloc[0]) for o in xx]
        rect=ax.bar(xx-.39+j*w+w/2,vals,width=w*.91,label=SHORT[m],color=COL[m])
        for x,y in zip(rect,vals):
            if y>0:ax.text(x.get_x()+x.get_width()/2,y+.17,str(y),ha='center',va='bottom',fontsize=9)
    ax.set_xticks(xx, [f'Order {v}' for v in xx]);ax.set_ylim(0,13.25);ax.set_yticks([0,3,6,9,12]);
    ax.set_ylabel('Trials meeting test NMSE ≤ 10⁻³ (out of 12)');ax.set_title(title,fontweight='bold');
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    ax.legend(loc='upper left',ncol=3,fontsize=9,frameon=False)
    savefig(filename)

grouped_success('wrong',['PATT-rank-exchange','Tucker-rank-search-8','Tucker-ALS','Tucker-HOSVD+ALS'],
    'chart_wrong_rank_success.png','Wrong-rank initialization · matched 25% observations')
grouped_success('correct',['PATT-rank-exchange','Tucker-ALS','Tucker-ALS-24','Tucker-HOSVD+ALS'],
    'chart_correct_rank_success.png','Correct-rank initialization · matched 25% observations')

fig,ax=plt.subplots(figsize=(10,4.8));xx=np.arange(3,7)
for m in ['PATT-rank-exchange','Tucker-rank-search-8','Tucker-ALS','Tucker-HOSVD+ALS']:
    d=summary[(summary.rank_setup=='wrong')&(summary.method==m)].set_index('order')
    ax.plot(xx,d.loc[xx].median_test_nmse,'o-',label=SHORT[m],color=COL[m],linewidth=2,markersize=7)
ax.axhline(1e-3,color='#b5443c',ls='--',lw=1.3,label='NMSE threshold 10⁻³')
ax.set_yscale('log');ax.set_xticks(xx,[f'Order {v}' for v in xx]);ax.set_ylabel('Median held-out test NMSE (log scale)')
ax.set_title('Wrong-rank initializations: reconstruction accuracy',fontweight='bold')
ax.grid(alpha=.19,which='both');ax.legend(fontsize=9,ncol=3,frameon=False,loc='lower left')
savefig('chart_wrong_rank_nmse.png')

fig,ax=plt.subplots(figsize=(10,4.8));xx=np.arange(3,7)
for m in ['PATT-rank-exchange','Tucker-rank-search-8','Tucker-ALS','Tucker-ALS-24']:
    d=summary[(summary.rank_setup=='wrong')&(summary.method==m)].set_index('order')
    ax.plot(xx,d.loc[xx].median_total_s,'o-',label=SHORT[m],color=COL[m],linewidth=2,markersize=7)
ax.set_yscale('log');ax.set_xticks(xx,[f'Order {v}' for v in xx]);ax.set_ylabel('Median total CPU runtime (seconds, log scale)')
ax.set_title('End-to-end runtime · wrong-rank tests',fontweight='bold')
ax.grid(alpha=.19,which='both');ax.legend(fontsize=9,ncol=2,frameon=False,loc='upper left')
savefig('chart_wrong_rank_runtime.png')

fig,ax=plt.subplots(figsize=(10,4.8));colors={'mcar':'#1966ae','block':'#e59a42'}
p=raw[(raw.rank_setup=='wrong')&(raw.method=='PATT-rank-exchange')]
xx=np.arange(3,7);w=.3
for j,mask in enumerate(['mcar','block']):
    vals=[int(((p.order==o)&(p['mask']==mask)&(p.test_nmse<=1e-3)).sum()) for o in xx]
    bars=ax.bar(xx + (j-.5)*w,vals,width=.29,color=colors[mask],label=mask.upper() if mask=='mcar' else 'Block missingness')
    for b,y in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,y+.07,str(y),ha='center',fontsize=10)
ax.set_ylim(0,6.8);ax.set_xticks(xx,[f'Order {v}' for v in xx]);ax.set_yticks([0,1,2,3,4,5,6]);ax.set_ylabel('Successes out of 6 paired seeds');ax.grid(axis='y',alpha=.18)
ax.set_title('PATT rank-exchange success by mask geometry',fontweight='bold');ax.legend(frameon=False)
savefig('chart_mask_breakdown.png')

pri=pd.read_csv(ROOT/'results_certified_priority.csv')
fig,ax=plt.subplots(figsize=(9.2,4.7));inds=np.arange(2);width=.3
for j,method in enumerate(['original_full_screen','certified_priority']):
    gp=pri[pri.method==method].groupby('order').agg(med=('wall_s','median'),solves=('n_expanded_core_fits','median'))
    vals=[gp.loc[o,'med'] for o in [4,5]]
    bars=ax.bar(inds+(j-.5)*width,vals,width,label='Full shortlist' if j==0 else 'Certificate priority',color='#868d98' if j==0 else '#1966ae')
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.025,f'{v:.3f}s',ha='center',fontsize=10)
ax.set_xticks(inds,['Order 4','Order 5']);ax.set_ylim(0,1.15);ax.set_ylabel('Median wall time (seconds)');ax.set_title('Certificate-priority ablation (12 paired trials per order)',fontweight='bold');ax.grid(axis='y',alpha=.18);ax.legend(frameon=False)
savefig('chart_certificate_priority_runtime.png')

basin=pd.read_csv(ROOT/'results_post_exchange_basin.csv')
fig,ax=plt.subplots(figsize=(9.3,4.6))
fracs=sorted(basin.p.unique());bar=ax.bar([f'{p:.0%}' for p in fracs],[(basin[(basin.p==p)&(basin.test_nmse_after10<=1e-3)].shape[0]) for p in fracs],width=.55,color=['#a0a8b2','#3b80bb','#1966ae'])
for b in bar:ax.text(b.get_x()+b.get_width()/2,b.get_height()+.15,f'{int(b.get_height())}/12',ha='center',fontsize=12)
ax.set_ylim(0,13);ax.set_yticks([0,3,6,9,12]);ax.set_ylabel('Test NMSE successes after 10 ALS sweeps');ax.set_xlabel('Observed fraction')
ax.set_title('Rank correction is not sufficient at very low density',fontweight='bold');ax.grid(axis='y',alpha=.17)
savefig('chart_postexchange_refinement.png')

real=pd.read_csv(ROOT/'results_real_digits.csv').sort_values('median_nmse')
fig,ax=plt.subplots(figsize=(9.8,4.7));ax.barh(real.method[::-1],real.median_nmse[::-1],color=['#777f8d' if 'PATT' not in m else '#1966ae' for m in real.method[::-1]])
ax.set_xlabel('Median held-out NMSE (lower is better)');ax.set_title('Real digits: constrained Tucker budget (0/12 successes for every method)',fontweight='bold')
for i,v in enumerate(real.median_nmse[::-1]):ax.text(v+.002,i,f'{v:.4f}',va='center',fontsize=10)
ax.set_xlim(0,.45);ax.grid(axis='x',alpha=.17)
savefig('chart_real_digits.png')

stab=pd.read_csv(ROOT/'results_spectral_stability.csv');ok=stab[stab.targeted_eligible==1]
fig,ax=plt.subplots(figsize=(6.5,5.5));ax.scatter(ok.observed_angle_deg,ok.targeted_angle_bound_deg,alpha=.70,s=29,color='#1966ae')
top=max(ok.targeted_angle_bound_deg.max(),ok.observed_angle_deg.max());ax.plot([0,top],[0,top],'--',color='#b5443c',label='Bound = actual angle')
ax.set_xlim(0,top*1.04);ax.set_ylim(0,top*1.04);ax.set_xlabel('Observed angle (degrees)');ax.set_ylabel('Targeted theoretical upper bound (degrees)');ax.set_title(f'Targeted receiver stability · {len(ok)}/72 applicable',fontweight='bold');ax.legend(frameon=False);ax.grid(alpha=.14)
savefig('chart_receiver_stability.png')

# Explain the data fields and verify every core count used in the README
assert len(raw)==432
wrong=raw[(raw.rank_setup=='wrong')&(raw.method=='PATT-rank-exchange')]
assert len(wrong)==48 and int((wrong.test_nmse<=1e-3).sum())==32
assert len(pri)==48 and int((pri.test_nmse<=1e-3).sum())==46
assert len(basin)==36 and int(basin.reached_target.sum())==24
print('Charts rendered. Primary=432; PATT wrong-rank 32/48; priority 46/48; basin 24/36.')
