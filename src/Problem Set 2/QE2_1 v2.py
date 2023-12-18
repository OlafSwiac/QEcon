import numpy as np
from scipy.stats import betabinom
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# Defining class of the model
class Model:
    def __init__(self,n=50, w_min=10, w_max=100, a=200, b=100, alfa=0.1, beta=0.96,
                            c=0.1, l=0.1, ro=0.02, H_vals = np.linspace(1,4,7)):
        self.n=n
        self.w_min=w_min
        self.w_max=w_max
        self.w_vals = np.linspace(w_min, w_max, 51)
        self.a = a
        self.b = b
        self.alfa = alfa
        self.beta = beta
        self.c = c
        self.l = l
        self.ro = ro
        self.H_vals = H_vals
        self.employments = [0, 1]
        self.fi = betabinom(self.n, self.a, self.b)
        self.v_init = np.zeros(shape=(len(self.w_vals), len(self.H_vals), len(self.employments)))

        # Defining a list of all states (combinations of (w_lvl, H_lvl, employment))
        self.states = []
        for i_w, w in enumerate(self.w_vals):
            for i_H, H in enumerate(self.H_vals):
                for employment in self.employments:
                    self.states.append({'w': i_w, 'H': i_H, 'employment': employment})
                    self.v_init[i_w, i_H, employment] = w * H / (1 - self.beta)

# Bellmann operator (m here is abbreviation for 'model')
def T(v, m):
    # Defining a result matrix representing new v
    v_res = np.zeros(shape=v.shape)
    
    max_H = len(m.H_vals) - 1

    # Assigning a result to each field of the matrix (corresponding to some state)
    for state in m.states:
        probs = betabinom.pmf(np.linspace(0, m.n, m.n+1), m.n, m.a, m.b)
        vec1 = m.alfa * v[:, min(state['H'] + 1, max_H), 0] + (1 - m.alfa) * v[:, state['H'], 0]
        continue_work_ev = m.beta * (1 - m.ro) * (m.alfa * v[state['w'], min(state['H'] + 1, max_H), 1] + (1 - m.alfa) * v[state['w'], state['H'], 1])
        lost_job_ev = m.beta * m.ro * np.dot(probs, vec1)
        if state['employment'] == 0:
            vec2 = m.l * v[:, max(state['H'] - 1, 0), 0] + (1 - m.l) * v[:, state['H'], 0]
            v_res[state['w'], state['H'], 0] = max(m.w_vals[state['w']] * m.H_vals[state['H']] + continue_work_ev + lost_job_ev,
                                                   m.c + np.dot(probs, vec2))
        elif state['employment'] == 1:
            v_res[state['w'], state['H'], 1] = m.w_vals[state['w']] * m.H_vals[state['H']] + continue_work_ev + lost_job_ev

    return v_res

# Value function iteration
def vfi(model, maxiter=10000, tol=1e-3, verbose = False):

    v_init = model.v_init
    error = tol + 1
    iters = 1

    v = v_init
    v_history = [v_init]

    while error > tol and iters < maxiter:

        v_new = T(v, model)
        error = np.max(np.abs(v_new - v))
        v_history.append(v_new)
        v = v_new
        if verbose:
            print('Iter {}. Error {}'.format(iters, error))
        iters += 1

    return v, iters, error, v_history


model1 = Model()
model2 = Model(alfa=0.2)

v1, iters1, error1, v_history1 = vfi(model1)
v2, iters2, error2, v_history2 = vfi(model2)

# Plot configuration for result presentation
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

for i, H in enumerate(model1.H_vals):
    ax1.plot(model1.w_vals, v1[:, i, 0])
    ax1.title.set_text('Model with alpha = 0.1')
    ax1.set_xlabel('w')
    ax1.set_ylabel('V value')
    ax1.grid(color=(0.8, 0.8, 0.8))
for i, H in enumerate(model2.H_vals):
    ax2.plot(model2.w_vals, v2[:, i, 0])
    ax2.title.set_text('Model with alpha = 0.2')
    ax2.set_xlabel('w')
    ax2.set_ylabel('V value')
    ax2.grid(color=(0.8, 0.8, 0.8))

fig.show()


