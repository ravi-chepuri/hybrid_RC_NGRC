import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from . import prediction_analysis


COLORS = ['#000000', '#004488', '#DDAA33', '#BB5566']  #https://personal.sron.nl/~pault/ #00264D for darker blue

def representative_trajectories(t_pred, actual, predictions_RC, predictions_NGRC, predictions_hyb, 
                                     component=0, plot_timesteps=10000, colors=COLORS, fname='2a.png',
                                     system='Lorenz'):
    
    VPT_hyb  = prediction_analysis.valid_pred_time(predictions_hyb,  actual, t_pred)
    VPT_RC   = prediction_analysis.valid_pred_time(predictions_RC,   actual, t_pred)
    VPT_NGRC = prediction_analysis.valid_pred_time(predictions_NGRC, actual, t_pred)

    fig, axs = plt.subplots(3, 1, figsize=(4*1.5, 3*1.5), sharex=True, sharey=False)

    for ax in axs:
        ax.plot(t_pred[:plot_timesteps], actual[component, :plot_timesteps], color=colors[0], ls='--')
        if system=='Lorenz': ax.set_xlim((0, 15))

    axs[0].plot(t_pred[:plot_timesteps], predictions_RC[component, :plot_timesteps], color=colors[1], label='RC')
    axs[1].plot(t_pred[:plot_timesteps], predictions_NGRC[component, :plot_timesteps], color=colors[2], label='NGRC')
    axs[2].plot(t_pred[:plot_timesteps], predictions_hyb[component, :plot_timesteps], color=colors[3], label='Hybrid RC-NGRC')

    axs[0].axvline(VPT_RC, ls='-.', color='grey')
    axs[1].axvline(VPT_NGRC, ls='-.', color='grey')
    axs[2].axvline(VPT_hyb, ls='-.', color='grey')

    for ax in axs: ax.legend(loc='upper right', framealpha=1.)

    fig.text(0.04, 0.5, f'{system} system $x$ variable', va='center', rotation='vertical')
    plt.xlabel('Prediction time (Lyapunov times)')
    plt.suptitle(f'Autonomous predictions of {system} system\n(representative example)')

    plt.savefig(fname)


def valid_prediction_time_violins(VPTs_RC, VPTs_NGRC, VPTs_hyb,
                                       colors=COLORS, fname='2b.png'):
    
    fig = plt.figure(figsize=(2.5*1.5, 3*1.5))
    violin = plt.violinplot([VPTs_RC, VPTs_NGRC, VPTs_hyb], showmedians=True, quantiles=[[0.25, 0.75]]*3)
    plt.xticks(ticks=[1,2,3], labels=['RC', 'NGRC', 'Hybrid RC-NGRC'])
    plt.ylabel('Valid prediction time (Lyapunov times)')
    plt.ylim(bottom=0)

    violin['bodies'][0].set_facecolor(colors[1])
    violin['bodies'][1].set_facecolor(colors[2])
    violin['bodies'][2].set_facecolor(colors[3])

    for pc in violin["bodies"]:
        pc.set_alpha(1)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cquantiles'):
        vp = violin[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)

    plt.title('Short term predictive power')

    plt.savefig(fname)


def phase_space_trajectories_3d(actual, predictions_RC, predictions_NGRC, predictions_hyb, 
                                     plot_timesteps=10000, colors=COLORS, fname='3a.png'):
    
    fig, axs = plt.subplots(2, 2, figsize=(3*1.5, 3*1.5))

    for row in axs:
        for ax in row:
            ax.plot(actual[0, :plot_timesteps], actual[2, :plot_timesteps], color=colors[0], lw=0.2, alpha=0.5, ls='--')
            ax.set_xlabel('$x$', labelpad=-10)
            ax.set_ylabel('$z$', labelpad=-10)
            # ax.tick_params(direction='in', length=3, pad=1)
            ax.set_xticks([-2, 2])
            ax.set_yticks([-2, 2])
            ax.set_xlim([-2.7, 2.7])
            ax.set_ylim([-2.7, 2.7])

    axs[0, 0].set_title('True system')
    axs[0, 1].set_title('RC')
    axs[1, 0].set_title('NGRC')
    axs[1, 1].set_title('Hybrid RC-NGRC')

    axs[0, 1].plot(predictions_RC[0, :plot_timesteps], predictions_RC[2, :plot_timesteps], color=colors[1], lw=0.5, alpha=1.)
    axs[1, 0].plot(predictions_NGRC[0, :plot_timesteps], predictions_NGRC[2, :plot_timesteps], color=colors[2], lw=0.5, alpha=1.)
    axs[1, 1].plot(predictions_hyb[0, :plot_timesteps], predictions_hyb[2, :plot_timesteps], color=colors[3], lw=0.5, alpha=1.)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)


def power_spectra_trajectories(actual, predictions_RC, predictions_NGRC, predictions_hyb, 
                                     component=2, nperseg=1000, colors=COLORS, fname='3b.png'):
    
    fig = plt.figure(figsize=(3.25*1.5, 3*1.5))

    f_actual, Pxx_den_actual = welch(actual[component], nperseg=nperseg)
    plt.semilogy(f_actual, Pxx_den_actual, color=colors[0], ls='dashed', label='True system')

    f_RC, Pxx_den_RC = welch(predictions_RC[component], nperseg=nperseg)
    plt.semilogy(f_RC, Pxx_den_RC, color=colors[1], ls='dotted', label='RC')

    f_NGRC, Pxx_den_NGRC = welch(predictions_NGRC[component], nperseg=nperseg)
    plt.semilogy(f_NGRC, Pxx_den_NGRC, color=colors[2], ls='dashdot', label='NGRC')

    f_hyb, Pxx_den_hyb = welch(predictions_hyb[component], nperseg=nperseg)
    plt.semilogy(f_hyb, Pxx_den_hyb, color=colors[3], ls='solid', label='Hybrid RC-NGRC')

    plt.xlabel('Frequency')
    plt.ylabel('Power spectral density')
    plt.title('Power spectra of autonomous predictions')
    plt.xlim((0, 0.3))
    plt.ylim((1e-3, 1e3))

    plt.legend()

    plt.tight_layout()
    plt.savefig(fname)


def VPT_vs_hyperparam(reservoir_sizes, trials, RC_data, RC_data_std, NGRC_data, NGRC_data_std, 
                               hyb_data, hyb_data_std, xlabel='', xscale=None, ylim=None, ls='none', linewidth=1,
                               colors=COLORS, fname='4.png'):
    
    plt.figure(figsize=(3*1.5, 2.5*1.5))
    
    plt.errorbar(reservoir_sizes, RC_data, yerr=RC_data_std/np.sqrt(trials), color=colors[1], marker='s', 
                 markersize=4, ls=ls, linewidth=linewidth, label='RC')
    plt.errorbar(reservoir_sizes, NGRC_data, yerr=NGRC_data_std/np.sqrt(trials), color=colors[2], marker='D', 
                 markersize=4, ls=ls, linewidth=linewidth, label='NGRC')
    plt.errorbar(reservoir_sizes, hyb_data, yerr=hyb_data_std/np.sqrt(trials), color=colors[3], marker='o', 
                 markersize=4, ls=ls, linewidth=linewidth, label='Hybrid RC-NGRC')

    plt.xlabel(xlabel)
    plt.ylabel('Valid prediction time (Lyapunov times)')
    plt.legend()
    plt.ylim(bottom=0)

    if xscale: plt.xscale(xscale)
    if ylim: plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)