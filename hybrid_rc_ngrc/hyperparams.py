import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    # RC
    num_inputs : int = 3
    N : int = 50  # changed from 100
    degree : float = 10.
    radius : float = 0.9  # changed from 0.99
    leakage : float = 1.0
    bias : float = 0.5  # changed from 0.0
    sigma : float = 1.
    discard_transient_length : int = 1000  # warmup time steps
    beta : float = 1e-8  # changed from 1e-3
    activation_func : '...' = np.tanh
    # NGRC
    d : int = 3  # same as num_inputs
    k : int = 2
    s : int = 1
    # data
    dt : float = 0.06
    int_dt : float = 0.001  # integration time step
    train_length : int = 10000
    noise_variance : float = 1e-6  # square root is noise standard deviation
    prediction_steps : int = 10000
