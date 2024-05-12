import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt

from dysts.flows import Rossler

from context import hybrid_rc_ngrc


reservoir_sizes = np.linspace(50, 500, 19).astype(int)
trials = 64

RC_data         = np.zeros(reservoir_sizes.shape)
RC_data_std     = np.zeros(reservoir_sizes.shape)
NGRC_data       = np.zeros(reservoir_sizes.shape)
NGRC_data_std   = np.zeros(reservoir_sizes.shape)
hyb_data        = np.zeros(reservoir_sizes.shape)
hyb_data_std    = np.zeros(reservoir_sizes.shape)

for idx, reservoir_size in enumerate(tqdm(reservoir_sizes)):
    h = hybrid_rc_ngrc.Hyperparameters(N=reservoir_size, prediction_steps=5000, dt=0.1)
    VPT_results = np.array(Parallel(n_jobs=-1)(delayed(hybrid_rc_ngrc.do_prediction_trial)(Rossler, h, return_VPTs=True) 
                              for trial in range(trials)))
    VPT_results /= (1 / 0.151)  # lyapunov time from dysts package
    VPTs_RC, VPTs_NGRC, VPTs_hyb = VPT_results.T

    RC_data[idx] = VPTs_RC.mean()
    NGRC_data[idx] = VPTs_NGRC.mean()
    hyb_data[idx] = VPTs_hyb.mean()
    RC_data_std[idx] = VPTs_RC.std()
    NGRC_data_std[idx] = VPTs_NGRC.std()
    hyb_data_std[idx] = VPTs_hyb.std()

# since NGRC data does not depend on reservoir size, use only the first value for plotting
NGRC_data[:] = NGRC_data[0]
NGRC_data_std[:] = NGRC_data_std[0]

COLORS = hybrid_rc_ngrc.plotting.COLORS

plt.figure(figsize=(3*1.5, 2.5*1.5))

ls='solid'
linewidth=1

plt.axhline(NGRC_data[0], color=COLORS[2], ls='solid', linewidth=linewidth, label='NGRC')
plt.fill_between(np.linspace(-10000, 10000, 5), NGRC_data[0]-NGRC_data_std[0]/np.sqrt(trials), 
                 NGRC_data[0]+NGRC_data_std[0]/np.sqrt(trials), color=COLORS[2], alpha=0.25, label='NGRC')
    
plt.errorbar(reservoir_sizes, RC_data, yerr=RC_data_std/np.sqrt(trials), color=COLORS[1], marker='s', 
            markersize=4, ls=ls, linewidth=linewidth, label='RC')
plt.errorbar(reservoir_sizes, hyb_data, yerr=hyb_data_std/np.sqrt(trials), color=COLORS[3], marker='o', 
            markersize=4, ls=ls, linewidth=linewidth, label='Hybrid RC-NGRC')

plt.xlim((30, 520))


plt.xlabel('Number of nodes in reservoirs $N$')
plt.ylabel('Mean valid prediction time (Lyapunov times)')
plt.ylim(bottom=0)

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[2], (handles[0], handles[1]), handles[3]], ['RC', 'NGRC', 'Hybrid RC-NGRC'])

plt.tight_layout()
plt.title('Rossler')
plt.savefig('rossler_VPT_vs_size.png', dpi=300)
