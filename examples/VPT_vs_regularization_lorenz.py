import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt

from dysts.flows import Lorenz

from context import hybrid_rc_ngrc


betas = np.array([1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
trials = 64

RC_data         = np.zeros(betas.shape)
RC_data_std     = np.zeros(betas.shape)
NGRC_data       = np.zeros(betas.shape)
NGRC_data_std   = np.zeros(betas.shape)
hyb_data        = np.zeros(betas.shape)
hyb_data_std    = np.zeros(betas.shape)

for idx, beta in enumerate(tqdm(betas)):
    h = hybrid_rc_ngrc.Hyperparameters(beta=beta, prediction_steps=2000)
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

colors = hybrid_rc_ngrc.plotting.COLORS

plt.figure(figsize=(3*1.5, 2.5*1.5))

linewidth=1
ls='none'
xlabel=r'Regularization $\beta$'
    
plt.errorbar(betas, RC_data, yerr=RC_data_std/np.sqrt(trials), color=colors[1], marker='s', 
                markersize=4, ls=ls, linewidth=linewidth, label='RC')
plt.errorbar(betas, NGRC_data, yerr=NGRC_data_std/np.sqrt(trials), color=colors[2], marker='D', 
                markersize=4, ls=ls, linewidth=linewidth, label='NGRC')
plt.errorbar(betas, hyb_data, yerr=hyb_data_std/np.sqrt(trials), color=colors[3], marker='o', 
                markersize=4, ls=ls, linewidth=linewidth, label='Hybrid RC-NGRC')

plt.xlabel(xlabel)
plt.ylabel('Valid prediction time (Lyapunov times)')
plt.legend()
plt.ylim(bottom=0)

plt.xscale('log')
plt.ylim(bottom=0)

plt.tight_layout()
plt.savefig('VPT_vs_regularization.png', dpi=300)