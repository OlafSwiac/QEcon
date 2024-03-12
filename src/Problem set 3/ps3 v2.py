import numpy as np
from numpy import ndarray
import quantecon as qe
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        self.beta = 0.97
        self.gamma = 1
        self.r = 0.01
        self.fi = 0.1
        # Average wage: 60 000 USD / year
        self.w = 1
        self.tau = 1 / 15
        self.num_states = 7
        self.a_list = np.arange(0, 1, 0.001)
        self.a_list = np.exp(self.a_list) - 1 - self.fi
        self.z_list = np.linspace(0, 2, self.num_states)


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


def T(v, model):
    # Defining a result matrix representing new v
    v_res = np.zeros(shape=v.shape)
    c_res = np.zeros(shape=v.shape)

    for i, a in enumerate(model.a_list):
        for j, z in enumerate(model.z_list):
            c_s = np.maximum(z * model.w + (1+model.r)*a - model.a_list, 0.000001)
            if model.gamma < 1:
                v_s = (np.power(c_s, 1 - model.gamma) - 1) / (
                        1 - model.gamma) + model.beta * np.matmul(v, P[j, :])
            else:
                v_s = np.log(c_s) + model.beta * np.matmul(v, P[j, :])
            a_prim_ind = np.argmax(v_s)

            v_res[i, j] = v_s[a_prim_ind]
            c_res[i, j] = c_s[a_prim_ind]
    return v_res, c_res


# Value function iteration
def vfi(model, maxiter=10000, tol=1e-3, verbose=False):
    v_init = np.zeros(shape=(len(model.a_list), model.num_states))
    error = tol + 1
    iters = 1
    v = v_init
    v_history = [v_init]

    while error > tol and iters < maxiter:
        print(iters, error)
        v_new, c = T(v, model)
        error = np.max(np.abs(v_new - v))
        v_history.append(v_new)
        v = v_new
        if verbose:
            print('Iter {}. Error {}'.format(iters, error))
        iters += 1

    return v, iters, error, v_history, c

model = Model()
P = discretize_ar1(ar_values, model.num_states)

v, iters, error, v_history, c = vfi(model)

mc = qe.MarkovChain(P, model.z_list)
stationary = np.array(mc.stationary_distributions).flatten()

mpc = np.zeros(shape=v.shape)
for j in range(mpc.shape[1]):
        mpc[:, j] = (np.interp(model.a_list + model.tau/(1+model.r), model.a_list, c[:, j]) - c[:, j]) \
                    * (1+model.r) / model.tau

avg_mpc = np.mean(np.matmul(mpc, stationary))
a_len = len(model.a_list)
plt.scatter(model.a_list, np.matmul(mpc, stationary), alpha=0.1)
plt.vlines([model.a_list[int(a_len * 0.1)], model.a_list[int(a_len * 0.5)], model.a_list[int(a_len * 0.9)]], 0, 1)
plt.show()

ax = plt.figure().add_subplot(projection='3d')


print('\nAverage MPC (assuming asset level and productivity level are independent): {}\n'.format(avg_mpc))
print('10th percentile: {}'.format(np.dot(stationary, mpc[int(a_len * 0.1), :])))
print('50th percentile: {}'.format(np.dot(stationary, mpc[int(a_len * 0.5), :])))
print('90th percentile: {}'.format(np.dot(stationary, mpc[int(a_len * 0.9), :])))

print('\nRatio: {}'.format(np.sum(model.a_list) / a_len / np.dot(stationary, model.z_list)))
