import numpy as np
import math
from abc import ABC, abstractmethod


class BaseChromosome(ABC):

    @abstractmethod
    def calculate_real_value(self, bin_ind):
        pass


class Chromosome(BaseChromosome):
    def __init__(self, interval):
        # Options
        self.interval = interval

        # Real value
        self.real_value = np.random.uniform(self.interval[0], self.interval[1])

    def calculate_real_value(self, bin_ind):
        binary_string = ''.join([str(elem) for elem in bin_ind])
        return self.interval[0] + int(binary_string, 2) * (self.interval[1] - self.interval[0]) / (
            math.pow(2, bin_ind.size) - 1)

    def __add__(self, other):
        c = Chromosome(self.interval)
        c.real_value = self.real_value + other.real_value
        return c

    def __sub__(self, other):
        c = Chromosome(self.interval)
        c.real_value = self.real_value - other.real_value
        return c

    def __mul__(self, other):
        c = Chromosome(self.interval)
        c.real_value = self.real_value * other
        return c

    def __abs__(self):
        c = Chromosome(self.interval)
        c.real_value = abs(self.real_value)
        return c

    # to delete if centric parents not used
    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, np.int64):
            result = self.real_value / other
            return result
        else:
            raise TypeError("Unsupported operand type(s) for /: 'Member' and '{}'".format(type(other).__name__))
