import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from dysts.flows import Lorenz

from context import hybrid_rc_ngrc

X_COMPONENT_INDEX = 0

h = hybrid_rc_ngrc.Hyperparameters(num_inputs=1, d=1, N=50, dt=0.01, k=10, s=1)

trials = 100

VPT_results = np.array(Parallel(n_jobs=-1)(delayed(hybrid_rc_ngrc.do_prediction_trial)(Lorenz, h, return_VPTs=True, component=X_COMPONENT_INDEX) 
                              for trial in tqdm(range(trials))))
VPT_results /= hybrid_rc_ngrc.lorenz_lyap_time
VPTs_RC, VPTs_NGRC, VPTs_hyb = VPT_results.T

hybrid_rc_ngrc.plotting.valid_prediction_time_violins(VPTs_RC, VPTs_NGRC, VPTs_hyb, 
                                                        title='Lorenz $x$-component\nshort term predictive power', 
                                                        labels=['RC', 'NGRC\n($k=10$)', 'Hybrid RC-NGRC'], 
                                                        fname='2b_x.png')
hybrid_rc_ngrc.plotting.valid_prediction_time_violins(VPTs_RC, VPTs_NGRC, VPTs_hyb, 
                                                        title='Lorenz $x$-component\nshort term predictive power', 
                                                        labels=['RC', 'NGRC\n($k=10$)', 'Hybrid RC-NGRC'], 
                                                        fname='2b_x.eps')

mRC   = np.median(VPTs_RC)
mNGRC = np.median(VPTs_NGRC)
mHyb  = np.median(VPTs_hyb)
print(f'Median valid prediction times\nRC: {mRC:.2f}\nNGRC: {mNGRC:.2f}\nHybrid RC-NGRC: {mHyb:.2f}')