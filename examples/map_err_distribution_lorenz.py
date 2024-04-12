import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt

from dysts.flows import Lorenz

from context import hybrid_rc_ngrc


h = hybrid_rc_ngrc.Hyperparameters()

trials = 64

map_err_results = np.array(Parallel(n_jobs=-1)(delayed(hybrid_rc_ngrc.do_prediction_trial)(Lorenz, h, return_avg_map_err=True) 
                              for trial in tqdm(range(trials))))
# LOG10
# map_err_results = np.log10(map_err_results)
# LOG10
map_err_RC, map_err_NGRC, map_err_hyb, map_err_ctrl = map_err_results.T

# hybrid_rc_ngrc.plotting.valid_prediction_time_violins(map_err_RC, map_err_NGRC, map_err_hyb, map_err_ctrl, fname='test.png')

mRC   = np.median(map_err_RC)
mNGRC = np.median(map_err_NGRC)
mHyb  = np.median(map_err_hyb)
mCtrl  = np.median(map_err_ctrl)
print(f'Median map errors\nRC: {mRC:.3f}\nNGRC: {mNGRC:.3f}\nHybrid RC-NGRC: {mHyb:.3f}\nControl: {mCtrl:.3f}')


colors = hybrid_rc_ngrc.plotting.COLORS

fig = plt.figure(figsize=(3.5*1.5, 3*1.5))
violin = plt.violinplot([map_err_RC, map_err_NGRC, map_err_hyb, map_err_ctrl], showmedians=True, quantiles=[[0.25, 0.75]]*4)
plt.xticks(ticks=[1,2,3,4], labels=['RC', 'NGRC', 'Hybrid RC-NGRC', 'Control'])
plt.ylabel('Average normalized map error')

violin['bodies'][0].set_facecolor(colors[1])
violin['bodies'][1].set_facecolor(colors[2])
violin['bodies'][2].set_facecolor(colors[3])
violin['bodies'][3].set_facecolor(colors[0])

for pc in violin["bodies"]:
    pc.set_alpha(1)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cquantiles'):
    vp = violin[partname]
    vp.set_edgecolor('black')
    vp.set_linewidth(1)

# plt.ylim(bottom=0)
plt.yscale('log')

plt.title('Map errors')

plt.savefig('map_errors.png')