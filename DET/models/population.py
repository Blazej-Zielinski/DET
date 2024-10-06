import concurrent.futures
import numpy as np

from diffEvoLib.models.enums.optimization import OptimizationType
from diffEvoLib.models.member import Member


class Population:
    def __init__(self, interval, arg_num, size, optimization: OptimizationType):
        self.size = size
        self.members = None
        self.optimization = optimization

        # chromosome config
        self.interval = interval
        self.arg_num = arg_num

    def generate_population(self):
        self.members = np.array([Member(self.interval, self.arg_num) for _ in range(self.size)])

    @staticmethod
    def calculate_fitness(member, fitness_fun):
        args = member.get_chromosomes()
        return member, fitness_fun(args)

    def update_fitness_values(self, fitness_fun):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.calculate_fitness, member, fitness_fun) for member in self.members]

            for future in concurrent.futures.as_completed(futures):
                member, fitness_value = future.result()
                member.fitness_value = fitness_value

        # for member in self.members:
        #     member.calculate_fitness_fun(fitness_fun)

    def get_best_members(self, nr_of_members):
        # Get the indices that would sort the array based on the key function
        sorted_indices = np.argsort([member.fitness_value for member in self.members])
        # Use the sorted indices to sort the array
        sorted_array = self.members[sorted_indices]
        return sorted_array[:nr_of_members]

    def mean(self):
        return np.mean([member.fitness_value for member in self.members])

    def std(self):
        return np.std([member.fitness_value for member in self.members])

    def __str__(self, population_label=""):
        output = f"Population{population_label}:"
        for m in self.members:
            output += f"\n{str(m)}"

        return output
