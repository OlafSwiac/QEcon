import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib


def get_distribution_plots(N: list):
    """
    Get plots of Poisson distribution with lambda = 1
    :param N: A list with the amount of Random variables plotted for each histogram
    """
    matplotlib.use('TkAgg')  # Problems with matplotlib library
    X = []
    for n in N:
        x = np.random.poisson(lam=1, size=(1000, n))
        x = (np.mean(x, axis=1) - 1) * n ** 0.5
        X.append(x)

    fig, axs = plt.subplots(2, 2)
    x_axis = np.linspace(-3, 3, 1000)

    axs[0, 0].hist(X[0], ec='black', density=True, bins=15)
    axs[0, 0].set_title('N = {}'.format(N[0]))
    axs[0, 0].plot(x_axis, norm.pdf(x_axis), color='red')

    axs[0, 1].hist(X[1], ec='black', density=True, bins=15)
    axs[0, 1].set_title('N = {}'.format(N[1]))
    axs[0, 1].plot(x_axis, norm.pdf(x_axis), color='red')

    axs[1, 0].hist(X[2], ec='black', density=True, bins=15)
    axs[1, 0].set_title('N = {}'.format(N[2]))
    axs[1, 0].plot(x_axis, norm.pdf(x_axis), color='red')

    axs[1, 1].hist(X[3], ec='black', density=True, bins=15)
    axs[1, 1].set_title('N = {}'.format(N[3]))
    axs[1, 1].plot(x_axis, norm.pdf(x_axis), color='red')

    plt.show()
