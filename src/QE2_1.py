import numpy as np
from scipy.stats import betabinom
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
alfa = 0.1
rho = 0.02
l = 0.1
H = [1, 1.5, 2, 2.5, 3, 3.5, 4]


def create_job_search_model(n=51, w_min=10, w_max=100, a=200, b=100, n_beta=51, beta=0.96, c=0.1):
    w_vals = np.linspace(w_min, w_max, n)
    fi = []
    for i in range(0, n_beta):
        fi.append(betabinom.pmf(i, n_beta, a, b))
    return n, w_vals, fi, beta, c


"""def get_h_new(h_old, n, w_vals, fi, beta, c):
    iter_sum = 0
    for i in range(0, n):
        iter_sum += fi[i] * max(h_old, (w_vals[i] * (1 - rho) + h_old * rho) / (1 - beta))
    return c + beta * iter_sum


h_vec = np.linspace(1, 4, 5)


def get_h_1(model, tol=10 ** (-8), maxiter=1000):
    h_init = model[4] / (1 - model[3])
    n = model[0]
    w_vals = model[1]
    fi = model[2]
    beta = model[3]
    c = model[4]

    h_old = float(h_init)
    h_new = h_old
    h_history = [h_old]
    error = tol + 1.0

    iter = 1

    while (error > tol) & (iter < maxiter):
        h_new = get_h_new(h_old, n, w_vals, fi, beta, c)
        error = abs(h_new - h_old)
        h_old = h_new
        h_history.append(h_old)
        iter += 1

    return h_new, iter, error, h_history


def get_v_from_h(model, h):
    n = model[0]
    w_vals = model[1]
    fi = model[2]
    beta = model[3]
    c = model[4]

    sigma = w_vals / (1 - beta)
    v = np.array(sigma) * np.array(w_vals) / (1 - beta) + (1 - np.array(sigma)) * h
    return v, sigma


my_model = create_job_search_model()
h, iter, error, h_history = get_h_1(my_model)
v, sigma = get_v_from_h(my_model, h)"""


def T(v, model):
    n = model[0]
    w_vals = model[1]
    fi = model[2]
    beta = model[3]
    c = model[4]

    results = []

    fi_v = 0
    for i in range(0, len(fi)):
        fi_v += v[i] * fi[i]
    for w in w_vals:
        results.append(max(w / (1 - beta), c + beta * fi_v))
    return results



def employed(w, beta, H, iter):
    if iter < 50:
        empl_better = employed(H+1, iter+1)
        empl_same = employed(H, iter+1)
        unempl = unemployed(H, iter+1)
        return (H * w + empl_better * (1 - rho) * alfa + empl_same * (1 - rho) * (1 - alfa) + unempl * rho) * beta
    return 0

def unemployed(fi, H, iter):
    if iter < 1000:
        empl = employed(H, iter + 1)
        unempl_worse = employed(H-1, iter + 1)
        unempl_same = unemployed(H, iter + 1)
        w = fi[np.random.randint(0, len(fi) - 1)]
        return (c + ) * beta
    return 0

def get_policy(v, model):
    n = model[0]
    w_vals = model[1]
    fi = model[2]
    beta = model[3]
    c = model[4]

    fi_v = 0
    results = []

    for i in range(0, len(fi)):
        fi_v += v[i] * fi[i]
    for w in w_vals:
        results.append(w / (1 - beta) >= c + beta * fi_v)

    return results


def vfi(model, maxiter=1000, tol=10 ** (-8)):
    n = model[0]
    w_vals = model[1]
    fi = model[2]
    beta = model[3]
    c = model[4]

    v_init = np.array(w_vals) / (1 - beta)
    error = tol + 1.0
    iter = 1

    v = v_init
    v_history = [v_init]

    while (error > tol) & (iter < maxiter):
        v_new = T(v, model)
        error = max(abs(np.array(v_new) - v))
        v_history.append(v_new)
        v = v_new
        iter += 1

    sigma = get_policy(v, model)

    return v, sigma, iter, error, v_history


model = create_job_search_model()
v, sigma, iter, error, v_history = vfi(model)

plt.plot(v)
plt.show()
