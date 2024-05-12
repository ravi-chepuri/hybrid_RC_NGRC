import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from dysts.flows import Lorenz

from context import hybrid_rc_ngrc


h = hybrid_rc_ngrc.Hyperparameters()

trials = 100

VPT_results = np.array(Parallel(n_jobs=-1)(delayed(hybrid_rc_ngrc.do_prediction_trial)(Lorenz, h, return_VPTs=True) 
                              for trial in tqdm(range(trials))))
VPT_results /= hybrid_rc_ngrc.lorenz_lyap_time
VPTs_RC, VPTs_NGRC, VPTs_hyb = VPT_results.T

hybrid_rc_ngrc.plotting.valid_prediction_time_violins(VPTs_RC, VPTs_NGRC, VPTs_hyb, fname='2b.svg')

mRC   = np.median(VPTs_RC)
mNGRC = np.median(VPTs_NGRC)
mHyb  = np.median(VPTs_hyb)
print(f'Median valid prediction times\nRC: {mRC:.2f}\nNGRC: {mNGRC:.2f}\nHybrid RC-NGRC: {mHyb:.2f}')