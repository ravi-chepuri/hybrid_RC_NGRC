import numpy as np


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