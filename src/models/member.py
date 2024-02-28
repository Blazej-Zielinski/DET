import numpy as np
from src.models.chromosome import Chromosome


class Member:
    def __init__(self, interval: list, args_num: int):
        self.chromosomes = np.array([Chromosome(interval) for _ in range(args_num)])
        self.fitness_value = None
        self.interval = interval
        self.args_num = args_num

    def calculate_fitness_fun(self, fitness_fun):
        self.fitness_value = fitness_fun([chromosome.real_value for chromosome in self.chromosomes])
        return self.fitness_value

    def is_member_in_interval(self):
        for chromosome in self.chromosomes:
            if not (self.interval[0] <= chromosome.real_value <= self.interval[1]):
                return False
        return True

    def __str__(self):
        # new_line = '\n'
        # tab = '\t'
        return f"Member: [\n" \
               f"\t Real values [" \
               f"{''.join(str(chromosome.real_value) + '; ' for chromosome in self.chromosomes)}] \n" \
               f"\t Fitness value: {self.fitness_value}\n" \
               f"]"
               #f"\t Bin values: [\n" \
               #f" {''.join(tab + tab + str(chromosome.binary_value) + new_line for chromosome in self.chromosomes)}" \
               #f"\t ]\n" \

    def __add__(self, other):
        chromosomes = self.chromosomes + other.chromosomes
        new_member = Member(self.interval, self.args_num)
        new_member.chromosomes = chromosomes
        return new_member

    # to delete if centric parents not used
    def __sub__(self, other):
        if isinstance(other, float):
            result = self.chromosomes - other
            return result
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Member' and '{}'".format(type(other).__name__))

    # to delete if centric parents not used
    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, np.int64):
            result = self.chromosomes / other
            return result
        else:
            raise TypeError("Unsupported operand type(s) for /: 'Member' and '{}'".format(type(other).__name__))

    def __lt__(self, other):
        return self.fitness_value < other.fitness_value

    def __eq__(self, other):
        return self.fitness_value == other.fitness_value

    def __gt__(self, other):
        return self.fitness_value > other.fitness_value

    def __le__(self, other):
        return self.fitness_value <= other.fitness_value

    def __ne__(self, other):
        return self.fitness_value != other.fitness_value

    def __ge__(self, other):
        return self.fitness_value >= other.fitness_value

    def __abs__(self):
        # Implementation of the __abs__() method
        abs_chromosomes = abs(self.chromosomes)
        new_member = Member(self.interval, self.args_num)
        new_member.chromosomes = abs_chromosomes
        return new_member
