import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from dysts.flows import Lorenz

from context import hybrid_rc_ngrc


timesteps = np.linspace(0.01, 0.07, 13)
trials = 64

RC_data         = np.zeros(timesteps.shape)
RC_data_std     = np.zeros(timesteps.shape)
NGRC_data       = np.zeros(timesteps.shape)
NGRC_data_std   = np.zeros(timesteps.shape)
hyb_data        = np.zeros(timesteps.shape)
hyb_data_std    = np.zeros(timesteps.shape)

for idx, timestep in enumerate(tqdm(timesteps)):
    h = hybrid_rc_ngrc.Hyperparameters(dt=timestep, prediction_steps=2000)
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

hybrid_rc_ngrc.plotting.VPT_vs_hyperparam(timesteps, trials, RC_data, RC_data_std, NGRC_data, NGRC_data_std, 
                                    hyb_data, hyb_data_std, xlabel=r'Timestep $\tau$', ls='solid',
                                    fname='lorenz_VPT_vs_timestep.png')