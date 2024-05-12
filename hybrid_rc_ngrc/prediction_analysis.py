import numpy as np

from dysts import flows
from dysts.base import DynSys


def _normalized_rms_error(prediction, actual):
    """
    Normalized RMS error in prediction is
    $$ e_j = \frac{||\bf{x}_j^{pred} - \bf{\hat{x}}_j||}{\sqrt{\langle ||\bf{\hat{x}}_j||^2 \rangle}} $$
    where $||...||$ is the Euclidean norm, $\langle ... \rangle$ is an average over prediction interval 
    (Wikner et al. 2021, p. 14).
    """
    numerator = np.linalg.norm(prediction - actual, axis=0)
    denominator = np.sqrt(np.mean(np.linalg.norm(actual, axis=0)**2))
    return numerator / denominator


def valid_pred_time(prediction, actual, prediction_times, threshold=0.9):
    """
    Valid prediction time is the time at which normalized RMS error first exceeds a threshold $\kappa$
    (Wikner et al. 2021, p. 14).
    """
    err = _normalized_rms_error(prediction, actual)
    prediction_diverges = np.any(np.isnan(prediction), axis=0)
    try: 
        # find smallest index where the error is greater than the threshold, or the prediction diverges (np.nan)
        index = np.where(np.logical_or(err > threshold, prediction_diverges))[0][0]  
    except IndexError:
        raise Exception('Error never exceeds threshold')
    return prediction_times[index]


def diverge_time(prediction, prediction_times):
    """Return the time at which prediction first diverges"""
    try:
        diverge_timestep = np.where(np.any(np.isnan(prediction), axis=0))[0][0]
    except IndexError:
        return np.nan
    return prediction_times[diverge_timestep]


def _mean_err_persistence_forecast(training_data):
    next_data = np.roll(training_data, -1, axis=1)
    diffs = next_data[:, :-1] - training_data[:, :-1]
    return np.mean(np.linalg.norm(diffs, axis=0))


def _true_integration_function(system: DynSys, method='RK45', norm_means=None, norm_stds=None, return_fine=False):
    """Produces a function that integrates the true system evolution equations forward from a given initial condition
    for a given time, and returns the final state.
    If the data has been normalized to have mean 0 and std 1, provide the original mean and std vectors.
    """

    def integrate_forward(init_cond, time, int_dt=0.001):
        # check that integration time step is an integer divisor of the desired time step
        assert abs(time/int_dt - round(time/int_dt)) < 1e-12, f'desired time {time} must be an integer multiple of integration time step {int_dt}'
        int_steps = int(round(time/int_dt)) + 1

        model = system()
        model.dt = int_dt

        if norm_means is not None:
            unnormalized_init_cond = init_cond * norm_stds + norm_means
            model.ic = unnormalized_init_cond
        else:
            model.ic = init_cond

        with np.errstate(divide='ignore', invalid='ignore'):
            if system is flows.MackeyGlass:
                traj = model.make_trajectory(int_steps, method=method, d=1).T
            else:
                traj = model.make_trajectory(int_steps, method=method).T

        if return_fine:
            return traj

        if norm_means is not None:
            return (traj[:, -1] - norm_means) / norm_stds
        else:
            return traj[:, -1]
        
    return integrate_forward


def mean_normalized_map_error(prediction, training_data, system: DynSys, dt, int_dt=0.001, method='RK45', 
                              num_steps=None, norm_means=None, norm_stds=None):

    if num_steps is not None:
        prediction = prediction[:, :num_steps]

    integrate_forward = _true_integration_function(system, method=method, norm_means=norm_means, norm_stds=norm_stds)
    evolutions = np.array([integrate_forward(pred_point, dt, int_dt=int_dt) 
                           for i, pred_point in enumerate(prediction.T)]).T

    diffs = prediction[:, 1:] - evolutions[:, :-1]

    return np.mean(np.linalg.norm(diffs, axis=0) / _mean_err_persistence_forecast(training_data))