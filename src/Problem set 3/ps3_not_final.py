import numpy as np
from numpy import ndarray
from scipy.stats import betabinom
import matplotlib.pyplot as plt
import matplotlib
import quantecon as qe


class Model:
    def __init__(self):
        self.beta = 0.97
        self.gamma = 0.9
        self.r = 0.01
        self.fi = 0.1
        self.w = 1
        self.tau = 1
        self.a_list = np.arange(-self.fi, 1, 0.1).tolist()
        self.z_list = np.linspace(0.9, 1.1, 7)


def simulate_ar1(n, phi, sigma):
    ar_values = np.zeros(n)
    for t in range(1, n):
        ar_values[t] = phi * ar_values[t - 1] + np.random.normal(loc=0, scale=sigma)
    ar_values = ar_values + 1
    return ar_values


def discretize_ar1(ar_values, num_states):
    min_val = 0
    max_val = 2

    state_boundaries = np.linspace(min_val, max_val, num_states + 1)

    transition_matrix: ndarray = np.zeros((num_states, num_states))

    for i in range(num_states):
        for j in range(num_states):
            transition_matrix[i, j] = np.sum(
                (state_boundaries[i] <= ar_values[:-1]) & (ar_values[:-1] < state_boundaries[i + 1]) & (
                        state_boundaries[j] <= ar_values[1:]) & (ar_values[1:] < state_boundaries[j + 1]))

    for i in range(num_states):
        transition_matrix[i, :] = transition_matrix[i, :] / np.sum(transition_matrix[i, :])

    return transition_matrix


n = 1000000
phi = 0.95
sigma = 0.1

ar_values = simulate_ar1(n, phi, sigma)

num_states = 7

P = discretize_ar1(ar_values, num_states)


def get_c(v, a, z, a_prim, model):
    value = v[a][z][a_prim]
    for z_prim in range(v.shape[1]):
        value -= model.beta * P[z][z_prim] * v[a][z_prim][a_prim]

    # c = (value * (1 - model.gamma)) ** (1 / (1 - model.gamma))  # mozna jeszcze uzyc wzoru c + a_prim = ... (?)
    c = model.z_list[z] * model.w + (1 + model.r) * model.a_list[a] - model.a_list[a_prim]
    return c


def T(v, model):
    # jest indeksem
    # Defining a result matrix representing new v
    v_res = np.zeros(shape=v.shape)

    for a in range(v.shape[0]):
        for z in range(v.shape[1]):
            for a_prim in range(v.shape[0]):
                for z_prim in range(v.shape[1]):
                    v_res[a][z][a_prim] += model.beta * P[z][z_prim] * v[a][z_prim][a_prim] + \
                                           ((model.z_list[z] * model.w + (1 + model.r) * model.a_list[a] - model.a_list[
                                               a_prim]) ** (1 - model.gamma)) \
                                           / (1 - model.gamma)

    return v_res


# Value function iteration
def vfi(model, maxiter=10000, tol=1e-3, verbose=False):
    v_init = np.zeros(shape=(len(model.a_list), 7, len(model.a_list)))
    error = tol + 1
    iters = 1

    v = v_init
    v_history = [v_init]

    while error > tol and iters < maxiter:
        print(iters, error)
        v_new = T(v, model)
        error = np.max(np.abs(v_new - v))
        v_history.append(v_new)
        v = v_new
        if verbose:
            print('Iter {}. Error {}'.format(iters, error))
        iters += 1

    return v, iters, error, v_history


def get_a_prim(v, a, z, model):
    a_max = 0
    v_max = v[a][z][a_max]
    for a_prim in range(len(model.a_list)):
        if v_max < v[a][z][a_prim]:
            v_max = v[a][z][a_prim]
            a_max = a_prim

    return a_max


def MPC(a, z, c_value_grid, model):
    delta_a = model.a_list[a] + model.tau / (1 + model.r)
    index_delta_a = (np.abs(c_value_grid - delta_a)).argmin()
    if model.a_list[index_delta_a] > delta_a:
        index_delta_a_1 = index_delta_a
        index_delta_a_2 = min(index_delta_a + 1, len(model.a_list))
    else:
        index_delta_a_1 = max(index_delta_a - 1, 0)
        index_delta_a_2 = index_delta_a

    a_1 = model.a_list[index_delta_a_1]
    a_2 = model.a_list[index_delta_a_2]

    c_1 = c_value_grid[a_1][z]
    c_2 = c_value_grid[a_2][z]

    delta_c = (c_2 - c_1) / (a_2 - a_1) * delta_a + c_1 - a_1 * (c_2 - c_1) / (a_2 - a_1)

    mpc = (delta_c - c_value_grid[a][z]) / (model.tau / (1 + model.r))

    return mpc


model = Model()

c_value_grid = np.zeros(shape=(len(model.a_list), 7))
v, iters, error, v_history = vfi(model)
for a in range(c_value_grid.shape[0]):
    for z in range(c_value_grid.shape[1]):
        a_prim = get_a_prim(v, a, z, model)
        c_value_grid[a][z] = get_c(v, a, z, a_prim, model)

mc = qe.MarkovChain(P, model.z_list)
stationary = np.array(mc.stationary_distributions).flatten()

