import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from dysts.flows import Lorenz

from context import hybrid_rc_ngrc


noises = np.logspace(-10, -1, 19)
noises_std = np.sqrt(noises)
trials = 64

RC_data         = np.zeros(noises.shape)
RC_data_std     = np.zeros(noises.shape)
NGRC_data       = np.zeros(noises.shape)
NGRC_data_std   = np.zeros(noises.shape)
hyb_data        = np.zeros(noises.shape)
hyb_data_std    = np.zeros(noises.shape)

for idx, noise in enumerate(tqdm(noises)):
    h = hybrid_rc_ngrc.Hyperparameters(noise_variance=noise, prediction_steps=2000)
    VPT_results = np.array(Parallel(n_jobs=-1)(delayed(hybrid_rc_ngrc.do_prediction_trial)(Lorenz, h, return_VPTs=True) 
                              for trial in range(trials)))
    VPT_results /= hybrid_rc_ngrc.lorenz_lyap_time
    VPTs_RC, VPTs_NGRC, VPTs_hyb = VPT_results.T

    RC_data[idx] = VPTs_RC.mean()
    NGRC_data[idx] = VPTs_NGRC.mean()
    hyb_data[idx] = VPTs_hyb.mean()
    RC_data_std[idx] = VPTs_RC.std()
    NGRC_data_std[idx] = VPTs_NGRC.std()
    hyb_data_std[idx] = VPTs_hyb.std()

hybrid_rc_ngrc.plotting.VPT_vs_hyperparam(noises_std, trials, RC_data, RC_data_std, NGRC_data, NGRC_data_std, 
                                    hyb_data, hyb_data_std, xlabel='Noise std. dev. $\gamma$', xscale='log',
                                    fname='VPT_vs_noise_lorenz.eps')