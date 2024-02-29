from dysts.flows import Lorenz

from context import hybrid_rc_ngrc


h = hybrid_rc_ngrc.Hyperparameters()  # use "default" hyperparameters as listed in Table I of the paper

t_pred, actual, predictions_RC, predictions_NGRC, predictions_hyb = hybrid_rc_ngrc.do_prediction_trial(Lorenz, h)
t_pred /= hybrid_rc_ngrc.dyn_systems.lorenz_lyap_time

hybrid_rc_ngrc.plotting.representative_trajectories(t_pred, actual, predictions_RC, predictions_NGRC, predictions_hyb, 
                                          fname='2a.svg')
hybrid_rc_ngrc.plotting.phase_space_trajectories_3d(actual, predictions_RC, predictions_NGRC, predictions_hyb,
                                          fname='3a.png')
hybrid_rc_ngrc.plotting.power_spectra_trajectories(actual, predictions_RC, predictions_NGRC, predictions_hyb,
                                         fname='3b.svg')