from opfunu.utils.visualize import plot_3d, plot_latex_formula


class FunctionObjectOld:
    def __init__(self, func, bounds):
        self.func = func
        self.bounds = bounds

    def evaluate(self, x):
        return self.func(x)


class FunctionObject:
    def __init__(self, func_type, bounds, ndim):
        self.func_type = func_type
        self.bounds = bounds
        self.function = func_type.value(ndim=ndim)

    def evaluate(self, params):
        return self.function.evaluate(params)


def plot_fun(func, bounds, ndim):
    """
    plot_fun(operator.discus_func, [(-100, 100), (-100, 100)])
    :param func: FUNCTIONS.F12014
    :param bounds: [(-100, 100), (-100, 100)]
    :return:
    """
    function_obj = FunctionObject(func, bounds, ndim)
    plot_3d(function_obj)


def plot_fun_formula(func):
    plot_latex_formula(func.latex_formula)
