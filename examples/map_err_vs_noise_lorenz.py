import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt

from dysts.flows import Lorenz

from context import hybrid_rc_ngrc


# noises = np.logspace(-10, -1, 19)
noises = np.logspace(-10, -1, 10)
noises_std = np.sqrt(noises)
trials = 4

RC_data         = np.zeros(noises.shape)
RC_data_std     = np.zeros(noises.shape)
NGRC_data       = np.zeros(noises.shape)
NGRC_data_std   = np.zeros(noises.shape)
hyb_data        = np.zeros(noises.shape)
hyb_data_std    = np.zeros(noises.shape)

for idx, noise in enumerate(tqdm(noises)):
    h = hybrid_rc_ngrc.Hyperparameters(noise_variance=noise, prediction_steps=1500)
    map_err_results = np.array(Parallel(n_jobs=-1)(delayed(hybrid_rc_ngrc.do_prediction_trial)(Lorenz, h, return_avg_map_err=True) 
                              for trial in range(trials)))
    map_errs_RC, map_errs_NGRC, map_errs_hyb, map_errs_ctrl = map_err_results.T

    RC_data[idx] = map_errs_RC.mean()
    NGRC_data[idx] = map_errs_NGRC.mean()
    hyb_data[idx] = map_errs_hyb.mean()
    RC_data_std[idx] = map_errs_RC.std()
    NGRC_data_std[idx] = map_errs_NGRC.std()
    hyb_data_std[idx] = map_errs_hyb.std()

# hybrid_rc_ngrc.plotting.VPT_vs_hyperparam(noises_std, trials, RC_data, RC_data_std, NGRC_data, NGRC_data_std, 
#                                     hyb_data, hyb_data_std, xlabel='Noise std. dev. $\gamma$', xscale='log',
#                                     fname='VPT_vs_noise_lorenz.png')

colors = hybrid_rc_ngrc.plotting.COLORS

plt.figure(figsize=(3*1.5, 2.5*1.5))
    
plt.errorbar(noises, RC_data, yerr=RC_data_std/np.sqrt(trials), color=colors[1], marker='s', 
                markersize=4, ls='none', linewidth=1, label='RC')
plt.errorbar(noises, NGRC_data, yerr=NGRC_data_std/np.sqrt(trials), color=colors[2], marker='D', 
                markersize=4, ls='none', linewidth=1, label='NGRC')
plt.errorbar(noises, hyb_data, yerr=hyb_data_std/np.sqrt(trials), color=colors[3], marker='o', 
                markersize=4, ls='none', linewidth=1, label='Hybrid RC-NGRC')

plt.xlabel('Noise std. dev. $\gamma$')
plt.ylabel('Mean normalized map error')
plt.legend()

plt.xscale('log')
plt.yscale('log')

plt.tight_layout()
plt.savefig('map_err_vs_noise.png', dpi=300)