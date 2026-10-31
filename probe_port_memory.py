"""Runs one benchmark case in a fresh Python process and reports process RSS.
Measurements include Python runtime, BLAS and imported modules.
"""
import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[v]='1'
import sys,json,resource
from pathlib import Path
from run_matlab_ports_benchmark import run_one
if __name__=='__main__':
 path,method,setup=sys.argv[1:4]
 r=run_one(Path(path),method,setup,maxiter=120)
 r['peak_process_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 print(json.dumps({'source_mat':Path(path).name,'method':method,'rank_setup':setup,
                   'peak_process_rss_mib':r['peak_process_rss_mib'],'elapsed_s':r['total_s'],
                   'test_nmse':r['test_nmse']}))
