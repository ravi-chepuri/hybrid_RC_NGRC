import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from dysts.flows import Lorenz

from context import hybrid_rc_ngrc


betas = np.array([1e-8, 1e-5])
noise_vars = np.array([1e-6])
trials = 2

map_err_results = np.array(Parallel(n_jobs=-1)(delayed(hybrid_rc_ngrc.do_prediction_trial)(Lorenz, 
                                                                                           hybrid_rc_ngrc.hyperparams.Hyperparameters(beta=beta,
                                                                                                                                      noise_variance=noise_var), 
                                                                                           return_avg_map_err=True)
                                                                                        #    return_VPTs=True)
                              for beta in tqdm(betas)
                              for noise_var in noise_vars 
                              for trial in range(trials)))
map_err_results = np.reshape(map_err_results, (betas.size, noise_vars.size, trials, 4))
print(map_err_results.shape)
np.save('map_err_results.npy', map_err_results)


mean_map_err_results = np.mean(map_err_results, axis=2)

mean_map_err_RC = mean_map_err_results[:, :, 0]
mean_map_err_NGRC = mean_map_err_results[:, :, 1]
mean_map_err_hyb = mean_map_err_results[:, :, 2]
mean_map_err_ctrl = mean_map_err_results[:, :, 3]

print(mean_map_err_RC.shape)

min = np.min(mean_map_err_results)
max = np.max(mean_map_err_results)


# same scale
fig, axs = plt.subplots(1, 4, figsize=(19.2, 4.8))
for i in range(4):

    mean_map_err = mean_map_err_results[:, :, i]  # 0:RC, 1:NGRC, 2:hyb, 3:ctrl
    ax = axs[i]
    ax.imshow(mean_map_err, origin='lower', norm=LogNorm(vmin=min, vmax=max))
    ax.set_xticks(np.arange(0, mean_map_err.shape[1], 1))
    ax.set_xticklabels(noise_vars)
    ax.set_yticks(np.arange(0, mean_map_err.shape[0], 1))
    ax.set_yticklabels(betas)

axs[0].set_title('RC')
axs[1].set_title('NGRC')
axs[2].set_title('Hybrid')
axs[3].set_title('Control')

plt.savefig('map_err_heatmap_samescale.png', dpi=300)



# diff scale
fig, axs = plt.subplots(1, 4, figsize=(19.2, 4.8))
for i in range(4):

    mean_map_err = mean_map_err_results[:, :, i]  # 0:RC, 1:NGRC, 2:hyb, 3:ctrl
    ax = axs[i]
    ax.imshow(mean_map_err, origin='lower', norm=LogNorm())
    ax.set_xticks(np.arange(0, mean_map_err.shape[1], 1))
    ax.set_xticklabels(noise_vars)
    ax.set_yticks(np.arange(0, mean_map_err.shape[0], 1))
    ax.set_yticklabels(betas)

axs[0].set_title('RC')
axs[1].set_title('NGRC')
axs[2].set_title('Hybrid')
axs[3].set_title('Control')

plt.savefig('map_err_heatmap_diffscale.png', dpi=300)