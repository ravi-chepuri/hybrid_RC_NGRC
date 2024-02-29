import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import zscore

from dysts import flows
from dysts.base import DynSys, staticjit


LONG_TRAJECTORY_DIRECTORY = os.path.join(os.path.dirname(__file__), 'long_trajectories/')

lorenz_lyap_exponent = 0.9056
lorenz_lyap_time = 1 / lorenz_lyap_exponent


def random_init_cond(system: DynSys):
    sys_name = repr(system)[20:-2].split('.')[-1]
    try:
        data = np.loadtxt(os.path.join(LONG_TRAJECTORY_DIRECTORY, f'{sys_name}.txt'), delimiter=',')
    except FileNotFoundError as e:
        raise FileNotFoundError('Create a long trajectory with `long_trajectory.py` '
                                'in order to select a random initial condition.') from e
    traj = data[:, 1:].T
    sampled_point = traj[:, np.random.randint(0, traj.shape[1])]
    return sampled_point


def make_trajectory_with_random_init_cond(system: DynSys, num_timesteps, dt, int_dt=0.001, epsilon=1e-3, method='RK45'):

    # check that integration time step is an integer divisor of the desired time step
    assert abs(dt/int_dt - round(dt/int_dt)) < 1e-12, f'desired time step {dt} must be an integer multiple of integration time step {int_dt}'
    multiple = int(round(dt/int_dt))

    model = system()
    model.dt = int_dt

    init_cond = random_init_cond(system)
    perturbation = np.random.normal(loc=0., scale=epsilon, size=init_cond.shape)
    model.ic = init_cond + perturbation

    with np.errstate(divide='ignore', invalid='ignore'):
        if system is flows.MackeyGlass:
            traj = model.make_trajectory(int(num_timesteps*multiple*1.1), method=method, d=1).T
        else:
            traj = model.make_trajectory(int(num_timesteps*multiple*1.1), method=method).T
    
    traj = traj[:, -num_timesteps*multiple:]
    traj = traj[:, ::multiple]  # subsample to desired time step
    t = np.linspace(dt, num_timesteps*dt, num_timesteps)
    return traj, t


class DoubleScroll(DynSys):

    def __init__(self, **kwargs):
        pass

    @staticjit
    def _rhs(V1, V2, I, t):
        # constants
        R1 = 1.2
        R2 = 3.44
        R4 = 0.193
        beta = 11.6
        Ir = 2.25e-5
        # eqns
        dV = V1 - V2
        V1_dot = V1/R1 - dV/R2 - 2*Ir*np.sinh(beta*dV)
        V2_dot = dV/R2 + 2*Ir*np.sinh(beta*dV) - I
        I_dot = V2 - R4*I
        return V1_dot, V2_dot, I_dot
    
    def rhs(self, X, t):
        return self._rhs(*X.T, t)