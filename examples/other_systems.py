import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

from dysts import flows
from dysts.flows import Rossler, MackeyGlass

from context import hybrid_rc_ngrc


def measure_VPT_many_trials(system, h, trials=100, large_RC_size=None, is_lorenz=False):

    VPT_results = np.array(Parallel(n_jobs=-1)(delayed(hybrid_rc_ngrc.do_prediction_trial)(system, h, return_VPTs=True) 
                                for trial in tqdm(range(trials))))
    if is_lorenz: VPT_results /= hybrid_rc_ngrc.lorenz_lyap_time
    VPTs_RC, VPTs_NGRC, VPTs_hyb = VPT_results.T

    if large_RC_size:
        h_large_RC = deepcopy(h)
        h_large_RC.N = large_RC_size
        VPT_results = np.array(Parallel(n_jobs=-1)(delayed(hybrid_rc_ngrc.do_prediction_trial)(system, h_large_RC, return_VPTs=True) 
                                    for trial in tqdm(range(trials))))
        if is_lorenz: VPT_results /= hybrid_rc_ngrc.lorenz_lyap_time
        VPTs_large_RC, _, _ = VPT_results.T

        return VPTs_RC, VPTs_NGRC, VPTs_hyb, VPTs_large_RC
    
    return VPTs_RC, VPTs_NGRC, VPTs_hyb


N = 100
N_large = 1000

mg_dt = 0.3
mg_delay_timesteps = int(flows.MackeyGlass().tau / mg_dt)

systems_and_hyperparameters = {
    Rossler:                        hybrid_rc_ngrc.Hyperparameters(N=200, dt=0.10),
    hybrid_rc_ngrc.DoubleScroll:    hybrid_rc_ngrc.Hyperparameters(N=N, dt=0.03),
    MackeyGlass:                    hybrid_rc_ngrc.Hyperparameters(s=mg_delay_timesteps, N=200, dt=mg_dt, 
                                                                   num_inputs=1, d=1, int_dt=0.1,
                                                                   prediction_steps=50000)
}

systems_and_VPTs = {}

for system, h in systems_and_hyperparameters.items():
    print(f'{system.__name__} system')
    VPTs_RC, VPTs_NGRC, VPTs_hyb, VPTs_large_RC = measure_VPT_many_trials(system, h, large_RC_size=N_large)
    systems_and_VPTs[system] = (VPTs_RC, VPTs_NGRC, VPTs_hyb, VPTs_large_RC)


## Plotting ##

COLORS = hybrid_rc_ngrc.plotting.COLORS

fig, axs = plt.subplots(1, len(systems_and_hyperparameters), figsize=(6.5*1.5, 3*1.5))
for idx, system in enumerate(systems_and_VPTs):
    ax = axs[idx]

    N_smallRC = systems_and_hyperparameters[system].N

    VPTs_large_RC = systems_and_VPTs[system][3]
    ax.axhline(np.median(VPTs_large_RC), color='#00264D', ls=':')
    ax.fill_between(np.linspace(0, 4, 5), np.quantile(VPTs_large_RC, 0.25), np.quantile(VPTs_large_RC, 0.75), color='#00264D', alpha=0.25)

    violin = ax.violinplot(systems_and_VPTs[system][:3], showmedians=True, quantiles=[[0.25, 0.75]]*3)
    ax.set_xticks(ticks=[1,2,3], labels=[f'RC\n$N={N_smallRC}$', 'NGRC', 'Hybrid\nRC-NGRC'])

    violin['bodies'][0].set_facecolor(COLORS[1])
    violin['bodies'][1].set_facecolor(COLORS[2])
    violin['bodies'][2].set_facecolor(COLORS[3])

    for pc in violin["bodies"]:
        pc.set_alpha(1)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cquantiles'):
        vp = violin[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)


    dt = systems_and_hyperparameters[system].dt

    match system.__name__:
        case 'Rossler': name = 'Rössler'
        case 'DoubleScroll': name = 'Double Scroll'
        case 'Chua': name = 'Chua'
        case 'MackeyGlass': name = 'Mackey Glass'
    ax.set_title(name + fr' ($\tau = {dt}$)')

    ax.set_xlim((0.5, 3.5))

axs[0].set_ylabel('Valid prediction time')

plt.savefig('other_systems_intdt0p1.svg', dpi=300)