from Task2 import get_functions_root, print_task_2
from problem_1 import get_distribution_plots
from zad3 import get_max_point_from_funtion


def f(x):
    return (x + 1 + x ** 2) ** (1 / 3) - x


result = get_functions_root(f, x0=1, alpha=0, eps=0.00000001, maxiter=100)
print_task_2(result)

N = [5, 25, 100, 1000]
get_distribution_plots(N)


def h(x):
    return -abs(x) + 10


result_3, max_y = get_max_point_from_funtion(h, x0=5, lambd=0.01, maxiter=10000, delta=0.01, N_delta=100)
