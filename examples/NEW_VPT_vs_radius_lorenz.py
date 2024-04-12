import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt

from dysts.flows import Lorenz

from context import hybrid_rc_ngrc


radii = np.arange(0.05, 1.00, 0.05)
trials = 64

RC_data         = np.zeros(radii.shape)
RC_data_std     = np.zeros(radii.shape)
NGRC_data       = np.zeros(radii.shape)
NGRC_data_std   = np.zeros(radii.shape)
hyb_data        = np.zeros(radii.shape)
hyb_data_std    = np.zeros(radii.shape)

for idx, radius in enumerate(tqdm(radii)):
    h = hybrid_rc_ngrc.Hyperparameters(radius=radius, prediction_steps=2000)
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

# since NGRC data does not depend on reservoir radius, use only the first value for plotting
NGRC_data[:] = NGRC_data[0]
NGRC_data_std[:] = NGRC_data_std[0]

COLORS = hybrid_rc_ngrc.plotting.COLORS

plt.figure(figsize=(3*1.5, 2.5*1.5))

ls='solid'
linewidth=1

plt.axhline(NGRC_data[0], color=COLORS[2], ls='solid', linewidth=linewidth, label='NGRC')
plt.fill_between(np.linspace(-0.05, 1.05, 5), NGRC_data[0]-NGRC_data_std[0]/np.sqrt(trials), 
                 NGRC_data[0]+NGRC_data_std[0]/np.sqrt(trials), color=COLORS[2], alpha=0.25, label='NGRC')
    
plt.errorbar(radii, RC_data, yerr=RC_data_std/np.sqrt(trials), color=COLORS[1], marker='s', 
            markersize=4, ls=ls, linewidth=linewidth, label='RC')
# plt.errorbar(reservoir_sizes, NGRC_data, yerr=NGRC_data_std/np.sqrt(trials), color=COLORS[2], marker='D', 
#             markersize=4, ls=ls, linewidth=linewidth, label='NGRC')
plt.errorbar(radii, hyb_data, yerr=hyb_data_std/np.sqrt(trials), color=COLORS[3], marker='o', 
            markersize=4, ls=ls, linewidth=linewidth, label='Hybrid RC-NGRC')

# plt.xlim((25, 1025))
plt.xlim((-0.05, 1.05))


plt.xlabel(r'Reservoir spectral radius $\rho$')
plt.ylabel('Valid prediction time (Lyapunov times)')
# plt.legend()
plt.ylim(bottom=0)

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[2], (handles[0], handles[1]), handles[3]], ['RC', 'NGRC', 'Hybrid RC-NGRC'])

# order = [1,0,2]
# plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

plt.tight_layout()
plt.savefig('VPT_vs_radius.png', dpi=300)