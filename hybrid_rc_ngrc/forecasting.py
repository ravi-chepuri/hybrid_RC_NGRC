import numpy as np
from sklearn.linear_model import Ridge

NUMPY_MAX_FLOAT = np.finfo('float64').max
BOUND = np.sqrt(NUMPY_MAX_FLOAT)

## function to create NGRC representation vectors
def _NGRC_repr_vectors(data, d=3, k=2, s=1):
    # If data.shape[1] == k, constructs a single NGRC representation vector.
    # If data.shape[1] > k, constructs a matrix of NGRC representation vectors.
    
    assert data.shape[0] == d, 'd must match dimensionality of input'
    data_length = data.shape[1]

    warmup_length = s * (k - 1)
    delays = s * np.arange(1, k)
    delay_inputs = [data[:, warmup_length:]] + [data[:, warmup_length-delay:-delay] for delay in delays]
    delay_inputs = np.concatenate(delay_inputs, axis=0)

    N_const = 1
    N_lin = d * k
    N_nonlin = N_lin * (N_lin + 1) / 2
    N = int(N_const + N_lin + N_nonlin)

    repr_vector = np.zeros((N, data_length - warmup_length))
    repr_vector[0] = 1  # constant terms
    repr_vector[1:N_lin+1] = delay_inputs  # linear terms
    # nonlinear terms (up to quadratic)
    cnt = 0
    for factor1 in range(N_lin):
        for factor2 in range(factor1, N_lin):
            repr_vector[N_lin + 1 + cnt] = delay_inputs[factor1] * delay_inputs[factor2]  ##
            cnt += 1

    return repr_vector


## function to run open loop for the hybrid method
def create_training_repr_vectors(input, train_length, A, W_in, activation_func, leakage, bias, d=3, k=2, s=1,
                                 init_res_state=None):
    
    # RC
    N = A.shape[0]
    res_states = np.zeros((N, train_length))
    if init_res_state is not None:
        res_states[:, 0] = init_res_state
    for t in range(train_length - 1):
        res_states[:, t+1] = (1-leakage)*res_states[:, t] + leakage * activation_func(A @ res_states[:, t] + W_in @ input[:, t+1] + bias)

    # NGRC
    warmup_length = s * (k - 1)
    repr_vectors = _NGRC_repr_vectors(input[:, :train_length], d=d, k=k, s=s)  # a matrix containing representation matrix over all training timesteps (minus warmup timesteps)
    repr_vectors_padded = np.concatenate([np.zeros((repr_vectors.shape[0], warmup_length)), repr_vectors], axis=1)  # pad repr vectors at the beginning so that there are the same number of them as there are RC states
    
    hyb_repr_vectors = np.concatenate([res_states, repr_vectors_padded], axis=0)

    return hyb_repr_vectors, warmup_length


## function to fit output matrix of the hybrid model, optionally choosing to only fit one of the components (RC or NGRC)
def fit_output_matrix(hyb_repr_vectors, targets, regularization_param, N_RC, d=3, k=2, s=1, which='both'):
    N_NGRC = int(1 + d*k + d*k*(d*k+1)/2)
    assert N_RC + N_NGRC == hyb_repr_vectors.shape[0], 'dimension of hybrid representation vector does not match RC and NGRC size'

    clf = Ridge(alpha=regularization_param, fit_intercept=False, solver='cholesky')
    match which:
        case 'both':
            clf.fit(hyb_repr_vectors.T, targets.T)
            W_out = clf.coef_
        case 'RC':
            clf.fit(hyb_repr_vectors[:N_RC].T, targets.T)
            W_out_RC = clf.coef_
            W_out_NGRC = np.zeros((d, N_NGRC))
            W_out = np.concatenate([W_out_RC, W_out_NGRC], axis=1)
        case 'NGRC':
            clf.fit(hyb_repr_vectors[N_RC:].T, targets.T)
            W_out_RC = np.zeros((d, N_RC))
            W_out_NGRC = clf.coef_
            W_out = np.concatenate([W_out_RC, W_out_NGRC], axis=1)

    return W_out


## function to run closed loop for the hybrid method
def closed_loop_forecast(W_out, predict_length, A, W_in, training_hyb_repr_vectors, training_input, activation_func, leakage, bias, d=3, k=2, s=1):

    N_RC = A.shape[0]
    warmup_length = s * (k - 1)

    predictions = np.zeros((d, (warmup_length+1) + predict_length))
    predictions[:, :(warmup_length+1)] = training_input[:, -(warmup_length+1):]

    NGRC_repr_vector = _NGRC_repr_vectors(predictions[:, :(warmup_length+1)], d=d, k=k, s=s).flatten()
    RC_state = training_hyb_repr_vectors[:N_RC, -1]
    hyb_repr_vector = np.concatenate([RC_state, NGRC_repr_vector], axis=0)

    for t in range(predict_length):
        predictions[:, (warmup_length+1) + t] = W_out @ hyb_repr_vector  ##

        if np.any(np.abs(predictions[:, (warmup_length+1) + t]) > BOUND):  # check for numerically diverging predictions
            predictions[:, (warmup_length+1) + t :] = np.nan
            break

        RC_state = (1-leakage)*RC_state + leakage * activation_func(A @ RC_state 
                                                                    + W_in @ predictions[:, (warmup_length+1) + t] 
                                                                    + bias)
        NGRC_repr_vector = _NGRC_repr_vectors(predictions[:, 1+t: (warmup_length+1) + 1+t], d=d, k=k, s=s).flatten()
        hyb_repr_vector = np.concatenate([RC_state, NGRC_repr_vector], axis=0)

    return predictions[:, (warmup_length+1):]
