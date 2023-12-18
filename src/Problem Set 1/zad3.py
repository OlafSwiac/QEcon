import numpy as np


def get_max_point_from_funtion(f, x0, lambd, maxiter, delta, N_delta):
    accepted_points = [x0]
    accepted_values = [f(x0)]
    x_n = x0
    x_max = x0
    count = 0
    loop = 0
    sum_alfa = 0

    while (count < N_delta) & (loop < maxiter):

        # losowanie epsilona z rozkladu N(0,1)
        epsilon = np.random.normal(loc=0, scale=1)

        # liczymy x* i alfa
        x_star = x_n + lambd * epsilon
        alfa = min([1, f(x_star) / f(x_n)])

        # liczymy sume alf zeby potem zwrocic ich srednia
        sum_alfa += alfa

        # liczymy x_n+1
        if np.random.rand() <= alfa:
            x_n = x_star

        # patrzenie czy lapiemy sie do deltowego otoczenia
        if abs(x_max - x_n) < delta:
            count += 1
        else:
            count = 0

        # zapisywanie max wartosci
        if f(x_n) > f(x_max):
            x_max = x_n

        # dopisujemy x_n i f(x_n) do odpowiednich list
        accepted_points.append(x_n)
        accepted_values.append(f(x_n))

        # zwiekszamy obroty
        loop += 1

    # liczenie wyniku
    if loop >= maxiter:
        solution = 'NaN'
        f_solution = 'NaN'
    else:
        solution = x_max
        f_solution = f(solution)

    return (loop // maxiter, solution, f_solution, accepted_points, accepted_values, sum_alfa / loop)
