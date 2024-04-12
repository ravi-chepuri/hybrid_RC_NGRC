import numpy as np

from . import dyn_systems
from . import create_reservoir
from . import forecasting
from . import prediction_analysis
from .hyperparams import Hyperparameters

from dysts.base import DynSys


def do_prediction_trial(system: DynSys, h: Hyperparameters, 
                        attractor_means_and_stds=None,
                        return_VPTs=False, return_diverge_times=False, return_avg_map_err=False):

    traj, t = dyn_systems.make_trajectory_with_random_init_cond(system, h.train_length + h.prediction_steps, h.dt, 
                                                                int_dt=h.int_dt)
    train_data = traj[:, :h.train_length]
    predict_data = traj[:, h.train_length:]

    if attractor_means_and_stds is not None:
        # Normalize so that each component of a trajectory on the true attractor has mean 0 and std 1.
        attractor_means = np.expand_dims(attractor_means_and_stds[0], -1)
        attractor_stds = np.expand_dims(attractor_means_and_stds[1], -1)
        train_data = (train_data - attractor_means) / attractor_stds
        predict_data = (predict_data - attractor_means) / attractor_stds
    else:
        # Normalize components of training data to have mean 0 and std 1. Apply the same transformation to predict data
        train_mean = np.expand_dims(np.mean(train_data, axis=1), -1)
        train_std = np.expand_dims(np.std(train_data, axis=1), -1)
        train_data = (train_data - train_mean) / train_std
        predict_data = (predict_data - train_mean) / train_std

    noise = np.sqrt(h.noise_variance) * np.random.normal(loc=0., scale=1., size=train_data.shape)

    A = create_reservoir.make_reservoir(h.N, h.degree, h.radius)
    W_in = create_reservoir.make_input_matrix(h.num_inputs, h.N, h.sigma)

    train_hyb_repr_vectors, NGRC_warmup_length = forecasting.create_training_repr_vectors(train_data+noise, h.train_length, 
                                                                            A, W_in, h.activation_func, h.leakage,
                                                                            h.bias, d=h.d, k=h.k, s=h.s)
    
    train_hyb_repr_vectors_for_fit = train_hyb_repr_vectors[:, h.discard_transient_length:-1]
    train_targets = train_data[:, h.discard_transient_length+1:]

    train_hyb_repr_vectors_for_fit_NGRC = train_hyb_repr_vectors[:, NGRC_warmup_length:-1]
    train_targets_NGRC = train_data[:, NGRC_warmup_length+1:]

    W_out_hyb   = forecasting.fit_output_matrix(train_hyb_repr_vectors_for_fit, train_targets, h.beta, 
                                                           h.N, d=h.d, k=h.k, s=h.s, which='both')
    W_out_RC    = forecasting.fit_output_matrix(train_hyb_repr_vectors_for_fit, train_targets, h.beta, 
                                                           h.N, d=h.d, k=h.k, s=h.s, which='RC')
    W_out_NGRC  = forecasting.fit_output_matrix(train_hyb_repr_vectors_for_fit_NGRC, train_targets_NGRC, h.beta, 
                                                           h.N, d=h.d, k=h.k, s=h.s, which='NGRC')

    predictions_hyb     = forecasting.closed_loop_forecast(W_out_hyb,  h.prediction_steps, A, W_in, 
                                                        train_hyb_repr_vectors, train_data, h.activation_func, 
                                                        h.leakage, h.bias, d=h.d, k=h.k, s=h.s)
    predictions_RC      = forecasting.closed_loop_forecast(W_out_RC,   h.prediction_steps, A, W_in, 
                                                        train_hyb_repr_vectors, train_data, h.activation_func, 
                                                        h.leakage, h.bias, d=h.d, k=h.k, s=h.s)
    predictions_NGRC    = forecasting.closed_loop_forecast(W_out_NGRC, h.prediction_steps, A, W_in, 
                                                        train_hyb_repr_vectors, train_data, h.activation_func, 
                                                        h.leakage, h.bias, d=h.d, k=h.k, s=h.s)

    actual = predict_data[:, :h.prediction_steps]
    t_pred = np.linspace(0, h.prediction_steps * h.dt - h.dt, h.prediction_steps)

    if return_VPTs:
        VPT_RC      = prediction_analysis.valid_pred_time(predictions_RC, actual, t_pred)
        VPT_NGRC    = prediction_analysis.valid_pred_time(predictions_NGRC, actual, t_pred)
        VPT_hyb     = prediction_analysis.valid_pred_time(predictions_hyb, actual, t_pred)
        return VPT_RC, VPT_NGRC, VPT_hyb
    
    if return_diverge_times:
        diverge_time_RC      = prediction_analysis.diverge_time(predictions_RC, t_pred)
        diverge_time_NGRC    = prediction_analysis.diverge_time(predictions_NGRC, t_pred)
        diverge_time_hyb     = prediction_analysis.diverge_time(predictions_hyb, t_pred)
        return diverge_time_RC, diverge_time_NGRC, diverge_time_hyb
    
    if return_avg_map_err:
        norm_means = np.squeeze(attractor_means) if attractor_means_and_stds is not None else np.squeeze(train_mean)
        norm_stds = np.squeeze(attractor_stds) if attractor_means_and_stds is not None else np.squeeze(train_std)
        map_err_RC   = prediction_analysis.mean_normalized_map_error(predictions_RC, train_data, system, h.dt, 
                                                                   int_dt=h.int_dt, method='RK45', num_steps=1000,
                                                                   norm_means=norm_means, norm_stds=norm_stds)
        map_err_NGRC = prediction_analysis.mean_normalized_map_error(predictions_NGRC, train_data, system, h.dt, 
                                                                   int_dt=h.int_dt, method='RK45', num_steps=1000,
                                                                   norm_means=norm_means, norm_stds=norm_stds)
        map_err_hyb  = prediction_analysis.mean_normalized_map_error(predictions_hyb, train_data, system, h.dt, 
                                                                   int_dt=h.int_dt, method='RK45', num_steps=1000,
                                                                   norm_means=norm_means, norm_stds=norm_stds)
        map_err_ctrl = prediction_analysis.mean_normalized_map_error(actual, train_data, system, h.dt, 
                                                                   int_dt=h.int_dt, method='RK45', num_steps=1000,
                                                                   norm_means=norm_means, norm_stds=norm_stds)
        return map_err_RC, map_err_NGRC, map_err_hyb, map_err_ctrl

    return t_pred, actual, predictions_RC, predictions_NGRC, predictions_hyb