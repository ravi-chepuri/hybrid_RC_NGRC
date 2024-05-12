import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

from dysts.flows import Lorenz

from context import hybrid_rc_ngrc


COLORS = hybrid_rc_ngrc.plotting.COLORS

# determine normalizations of the true attractor
traj, t = hybrid_rc_ngrc.dyn_systems.make_trajectory_with_random_init_cond(Lorenz, 100000, 0.06)
attractor_means = np.mean(traj, axis=1)
attractor_stds = np.std(traj, axis=1)

training_data_amounts = np.logspace(1, 5, 9).astype(int)
trials = 64
RC_transient_cutoff_frac = 0.25
t_sync_multiple = 10  # take approx 10 sync times so that "error" in reservoir state is on order of e^-10 \approx 5e-5

# for scaling regularization with training data length, beta = n_train * beta_tilde
beta_tilde = 1e-12

# hard case

dt = 0.06
N1 = 100
N = N1

t_sync = 2.131552332975897  # empirically determined sync time, approx. 2

RC_data         = np.zeros(training_data_amounts.shape)
RC_data_std     = np.zeros(training_data_amounts.shape)
NGRC_data       = np.zeros(training_data_amounts.shape)
NGRC_data_std   = np.zeros(training_data_amounts.shape)
hyb_data        = np.zeros(training_data_amounts.shape)
hyb_data_std    = np.zeros(training_data_amounts.shape)

for idx, training_data_amount in enumerate(tqdm(training_data_amounts)):
    training_data_amount = int(training_data_amount)
    RC_transient = int(RC_transient_cutoff_frac * training_data_amount)
    RC_transient = int(min(RC_transient, t_sync_multiple * t_sync))

    h = hybrid_rc_ngrc.Hyperparameters(train_length=training_data_amount, discard_transient_length=RC_transient,
                                       dt=dt, N=N, beta=beta_tilde*training_data_amount)
    VPT_results = np.array(Parallel(n_jobs=-1)(delayed(hybrid_rc_ngrc.do_prediction_trial)(Lorenz, h, return_VPTs=True,
                                                                            attractor_means_and_stds=(attractor_means, attractor_stds)) 
                              for trial in range(trials)))
    VPT_results /= hybrid_rc_ngrc.lorenz_lyap_time
    VPTs_RC, VPTs_NGRC, VPTs_hyb = VPT_results.T

    RC_data[idx] = VPTs_RC.mean()
    NGRC_data[idx] = VPTs_NGRC.mean()
    hyb_data[idx] = VPTs_hyb.mean()
    RC_data_std[idx] = VPTs_RC.std()
    NGRC_data_std[idx] = VPTs_NGRC.std()
    hyb_data_std[idx] = VPTs_hyb.std()

# easy case
    
dt = 0.01
N2 = 500
N = N2

t_sync = 2.2482971691228375  # empirically determined sync time, approx. 2

RC_data_2         = np.zeros(training_data_amounts.shape)
RC_data_std_2     = np.zeros(training_data_amounts.shape)
NGRC_data_2       = np.zeros(training_data_amounts.shape)
NGRC_data_std_2   = np.zeros(training_data_amounts.shape)
hyb_data_2        = np.zeros(training_data_amounts.shape)
hyb_data_std_2    = np.zeros(training_data_amounts.shape)

for idx, training_data_amount in enumerate(tqdm(training_data_amounts)):
    training_data_amount = int(training_data_amount)
    RC_transient = int(RC_transient_cutoff_frac * training_data_amount)
    RC_transient = int(min(RC_transient, t_sync_multiple * t_sync))

    h = hybrid_rc_ngrc.Hyperparameters(train_length=training_data_amount, discard_transient_length=RC_transient,
                        dt=dt, N=N, beta=beta_tilde*training_data_amount)

    VPT_results = np.array(Parallel(n_jobs=-1)(delayed(hybrid_rc_ngrc.do_prediction_trial)(Lorenz, h, return_VPTs=True,
                                                                            attractor_means_and_stds=(attractor_means, attractor_stds)) 
                              for trial in range(trials)))
    VPT_results /= hybrid_rc_ngrc.lorenz_lyap_time
    VPTs_RC, VPTs_NGRC, VPTs_hyb = VPT_results.T


    RC_data_2[idx] = VPTs_RC.mean()
    NGRC_data_2[idx] = VPTs_NGRC.mean()
    hyb_data_2[idx] = VPTs_hyb.mean()
    RC_data_std_2[idx] = VPTs_RC.std()
    NGRC_data_std_2[idx] = VPTs_NGRC.std()
    hyb_data_std_2[idx] = VPTs_hyb.std()

# plotting
    
fig, axs = plt.subplots(2, 1, figsize=(3*1.5, 3*1.5), sharex=True, sharey=True)

linewidth=1
    
axs[0].errorbar(training_data_amounts, RC_data, yerr=RC_data_std/np.sqrt(trials), color=COLORS[1], marker='s', 
            markersize=4, ls='solid', linewidth=linewidth, label=r'RC')
axs[0].errorbar(training_data_amounts, NGRC_data, yerr=NGRC_data_std/np.sqrt(trials), color=COLORS[2], marker='D', 
            markersize=4, ls='solid', linewidth=linewidth, label=r'NGRC')
axs[0].errorbar(training_data_amounts, hyb_data, yerr=hyb_data_std/np.sqrt(trials), color=COLORS[3], marker='o', 
            markersize=4, ls='solid', linewidth=linewidth, label='Hybrid')

axs[1].errorbar(training_data_amounts, RC_data_2, yerr=RC_data_std_2/np.sqrt(trials), color=COLORS[1], marker='s', 
            markersize=4, ls='solid', linewidth=linewidth, label=r'RC')
axs[1].errorbar(training_data_amounts, NGRC_data_2, yerr=NGRC_data_std_2/np.sqrt(trials), color=COLORS[2], marker='D', 
            markersize=4, ls='solid', linewidth=linewidth, label=r'NGRC')
axs[1].errorbar(training_data_amounts, hyb_data_2, yerr=hyb_data_std_2/np.sqrt(trials), color=COLORS[3], marker='o', 
            markersize=4, ls='solid', linewidth=linewidth, label='Hybrid')

axs[0].set_title(fr'$\tau = 0.06, N = {N1}$', y=1.0, pad=-14)
axs[1].set_title(fr'$\tau = 0.01, N = {N2}$', y=1.0, pad=-14)

axs[1].set_xlabel('Total number of training data time steps $n_{\mathrm{train}}$\n(including warm up/syncing)')
fig.text(0.04, 0.5, 'Mean valid prediction time (Lyapunov times)', va='center', rotation='vertical')
axs[1].legend(loc='lower right')
plt.ylim(bottom=0)
plt.xscale('log')

plt.tight_layout()

plt.savefig('lorenz_VPT_vs_ntrain.png', dpi=300)