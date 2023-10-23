from Task2 import get_functions_root, print_task_2


def f(x):
    return (x + 1 + x ** 2) ** (1 / 3) - x


result = get_functions_root(f, x0=1, alpha=0, eps=0.00000001, maxiter=100)

print_task_2(result)
