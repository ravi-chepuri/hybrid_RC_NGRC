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
map_err_RC, map_err_NGRC, map_err_hyb, map_err_ctrl = map_err_results.T

mRC   = np.mean(map_err_RC)
mNGRC = np.mean(map_err_NGRC)
mHyb  = np.mean(map_err_hyb)
mCtrl  = np.mean(map_err_ctrl)
print(f'Avg. mean normalized map errors\nRC: {mRC:.5f}\nNGRC: {mNGRC:.5f}\nHybrid RC-NGRC: {mHyb:.5f}\nControl: {mCtrl:.3f}')

seRC   = np.std(map_err_RC) / np.sqrt(trials)
seNGRC = np.std(map_err_NGRC) / np.sqrt(trials)
seHyb  = np.std(map_err_hyb) / np.sqrt(trials)
seCtrl  = np.std(map_err_ctrl) / np.sqrt(trials)
print(f'std. err. mean of normalized map errors\nRC: {seRC:.7f}\nNGRC: {seNGRC:.7f}\nHybrid RC-NGRC: {seHyb:.7f}\nControl: {seCtrl:.7f}')