import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from dysts.flows import MackeyGlass

from context import hybrid_rc_ngrc


# timesteps = np.linspace(0.05, 0.35, 13)
# timesteps = np.linspace(0.05, 1.0, 20)
timesteps = np.linspace(0.1, 0.9, 17)
trials = 64

RC_data         = np.zeros(timesteps.shape)
RC_data_std     = np.zeros(timesteps.shape)
NGRC_data       = np.zeros(timesteps.shape)
NGRC_data_std   = np.zeros(timesteps.shape)
hyb_data        = np.zeros(timesteps.shape)
hyb_data_std    = np.zeros(timesteps.shape)

for idx, timestep in enumerate(tqdm(timesteps)):
    mg_dt = timestep
    mg_delay_timesteps = int(MackeyGlass().tau / mg_dt)
    h = hybrid_rc_ngrc.Hyperparameters(s=mg_delay_timesteps, N=100, dt=mg_dt, 
                                                                   num_inputs=1, d=1,
                                                                   prediction_steps=5000)
    VPT_results = np.array(Parallel(n_jobs=-1)(delayed(hybrid_rc_ngrc.do_prediction_trial)(MackeyGlass, h, return_VPTs=True) 
                              for trial in range(trials)))
    VPT_results /= (1 / 0.0729)  # lyapunov time from dysts package
    VPTs_RC, VPTs_NGRC, VPTs_hyb = VPT_results.T
    RC_data[idx] = VPTs_RC.mean()
    NGRC_data[idx] = VPTs_NGRC.mean()
    hyb_data[idx] = VPTs_hyb.mean()
    RC_data_std[idx] = VPTs_RC.std()
    NGRC_data_std[idx] = VPTs_NGRC.std()
    hyb_data_std[idx] = VPTs_hyb.std()

hybrid_rc_ngrc.plotting.VPT_vs_hyperparam(timesteps, trials, RC_data, RC_data_std, NGRC_data, NGRC_data_std, 
                                    hyb_data, hyb_data_std, xlabel=r'Timestep $\tau$', ls='solid',
                                    fname='mackeyglass_VPT_vs_timestep.png')