from typing import Callable
from abc import ABC, abstractmethod


class FitnessFunctionBase(ABC):
    def __init__(self):
        self.name = ""
        self.function = None

    @abstractmethod
    def eval(self, params):
        pass


class FitnessFunction(FitnessFunctionBase):
    def __init__(self, func: Callable[..., float], custom_name=None):
        super().__init__()
        self.name = func.__name__ if custom_name is None else custom_name
        self.function = func

    def eval(self, params):
        return self.function(*params)


class FitnessFunctionOpfunu(FitnessFunctionBase):
    def __init__(self, func_type, ndim, custom_name=None):
        super().__init__()
        self.name = func_type.__name__ if custom_name is None else custom_name
        self.function = func_type(ndim=ndim)

    def eval(self, params):
        return self.function.evaluate(params)


class BenchmarkFitnessFunction(FitnessFunctionBase):
    def __init__(self, function):
        super().__init__()
        self.instance = function
        self.name = function.name

    def eval(self, params):
        return self.instance.evaluate_func(params)
