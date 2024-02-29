import os
import numpy as np

from dysts import flows
from dysts.base import DynSys
from .dyn_systems import DoubleScroll
from .dyn_systems import LONG_TRAJECTORY_DIRECTORY


def save_trajectory(system: DynSys, num_timesteps, dt=None, ic=None):
    model = system()
    if dt: model.dt = dt
    if ic: model.ic = ic
    traj = model.make_trajectory(num_timesteps)
    t = np.linspace(model.dt, num_timesteps * model.dt, num_timesteps)
    data = np.hstack((np.expand_dims(t, axis=-1), traj))  # 0th column is time, subsequent columns are data

    sys_name = repr(system)[20:-2]
    np.savetxt(os.path.join(LONG_TRAJECTORY_DIRECTORY, f'{sys_name}.txt'), data, delimiter=',')


def load_trajectory(filename):
    data = np.loadtxt(filename, delimiter=',')
    t = data[:, 0]
    traj = data[:, 1:].T
    print(t.shape, traj.shape)
    return traj, t


if __name__ == '__main__':
    systems = [flows.Lorenz,
               flows.MackeyGlass,
               flows.Rossler,
               DoubleScroll]
    num_timesteps = 65536
    for system in systems:
        if system is DoubleScroll:
            save_trajectory(system, num_timesteps, dt=0.25, ic=[0.126817, 0.904962, -0.770680])
        else:
            save_trajectory(system, num_timesteps)