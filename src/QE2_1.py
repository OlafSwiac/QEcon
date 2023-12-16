import numpy as np
from scipy.stats import betabinom
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
alfa = 0.1
rho = 0.02
l = 0.1
H = [1, 1.5, 2, 2.5, 3, 3.5, 4]
a = 200
b = 100
n = 50
H_begin = 0


def create_job_search_model(n=50, w_min=10, w_max=100, a=200, b=100, n_beta=51, beta=0.96, c=0.1):
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


def vU(w, H_old, model, iter):
    n = model[0]
    w_vals = model[1]
    fi = model[2]
    beta = model[3]
    c = model[4]

    if iter < 3:
        sum_wages_1 = 0
        for i in range(0, len(w_vals)):
            sum_wages_1 += (alfa * vU(w_vals[i], min(H_old + 1, len(H) - 1), model, iter + 1) + (1 - alfa) * vU(
                w_vals[i], H_old,
                model, iter + 1)) * fi[i]

        sum_wages_2 = 0
        for i in range(0, len(w_vals)):
            sum_wages_2 += (l * vU(w_vals[i], min(H_old - 1, 0), model, iter + 1) + (1 - l) * vU(w_vals[i], H_old,
                                                                                                 model, iter + 1)) * fi[
                               i]

        x1 = w * H[int(H_old)] + beta * (1 - rho) * (
                alfa * vE(w, min(H_old + 1, len(H) - 1), model, iter + 1) + (1 - alfa) * vE(w, H_old, model,
                                                                                               iter + 1)) + beta * rho * sum_wages_1
        x2 = c + beta * sum_wages_2
        return max(x1, x2)

    else:
        return 0


def vE(w, H_old, model, iter):
    n = model[0]
    w_vals = model[1]
    fi = model[2]
    beta = model[3]
    c = model[4]
    if iter < 3:
        sum_wages_1 = 0
        for i in range(0, len(w_vals)):
            sum_wages_1 += (alfa * vU(w_vals[i], min(H_old + 1, len(H) - 1), model, iter + 1) + (1 - alfa) * vU(
                w_vals[i], H_old,
                model, iter + 1)) * fi[i]

        return w * H[int(H_old)] + beta * (1 - rho) * (
                alfa * vE(w, min(H_old + 1, len(H) - 1), model, iter + 1) + (1 - alfa) * vE(w, H_old, model,
                                                                                            iter + 1)) + beta * rho * sum_wages_1
    else:
        return 0


def T(H_begin, model):
    n = model[0]
    w_vals = model[1]
    fi = model[2]
    beta = model[3]
    c = model[4]

    results = []

    for w in w_vals:
        results.append(vU(w, H_begin, model, 0))
        print(w)

    return results


def get_policy(v):
    results = []
    v_min = min(v)
    for v_val in v:
        results.append(v_val > v_min)
    return results


def vfi(model):
    n = model[0]
    w_vals = model[1]
    fi = model[2]
    beta = model[3]
    c = model[4]

    v = T(H_begin, model)
    sigma = get_policy(v)

    return v, sigma


model = create_job_search_model()
v, sigma = vfi(model)

plt.scatter(x=model[1] * H[H_begin], y=v)
plt.xlabel("wage * H_begin")
plt.show()
