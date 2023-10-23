def get_functions_root(f, x0: float, alpha: float, eps: float, maxiter: int) -> tuple:
    """
    Get the root of the function f.
    :param f: The function
    :param x0: Starting point
    :param alpha: Parameter of searching speed
    :param eps: Acceptable error
    :param maxiter: Maximum number of iterations
    :return: Tuple which holds: Boolean value indicating if we found the root, root value, f(root),
    difference |root - f(root) + root|, list of x values we searched through, ....
    """

    def g(x):
        return f(x) + x

    list_of_x = [x0]
    print(list_of_x)
    list_of_residuals = []
    for i in range(0, maxiter):
        x1 = g(x0)
        list_of_x.append(x1)
        list_of_residuals.append(abs(x1 - x0))
        if abs(x1 - x0) < eps * (1 + abs(x0)):
            return True, x1, f(x1), abs(x1 - g(x1)), list_of_x, list_of_residuals, i
        else:
            x0 = (1 - alpha) * g(x0) + alpha * x0

    return False, float('nan'), float('nan'), float('nan'), list_of_x, list_of_residuals, maxiter


def print_task_2(result: tuple) -> None:
    """
    Printing the results of Task 2
    :param result: Tuple with results
    :return: Printed values
    """
    if result[0]:
        print(
            'The root of funtion f was found.\n\nFound x = {root}, f(x) = {funtion_value}\n'
            'The difference between final two points: {diff}\nNumber of points: {num_points}, List of points: {points}'
            '\n''List of residuals: {residuals}'.format(
                root=result[1], funtion_value=result[2], diff=result[3], points=result[4], residuals=result[5],
                num_points=result[6]))
    else:
        print(
            'The root of funtion f was NOT found.\n\n'
            'The difference between final two points: {diff}\nNumber of points: {num_points}, List of points: {points}'
            '\n''List of residuals: {residuals}'.format(
                root=result[1], funtion_value=result[2], diff=result[3], points=result[4], residuals=result[5],
                num_points=result[6]))