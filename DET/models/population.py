import concurrent.futures
import numpy as np
from DET.models.enums.optimization import OptimizationType
from DET.models.member import Member

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
        return fitness_fun(args)

    def calculate_member(self, index, fitness_fun):
        self.members[index[0]].fitness_value = self.calculate_fitness(self.members[index[0]], fitness_fun)

    def update_fitness_values(self, fitness_fun, parallel_processing):
        worker = 1
        if parallel_processing is None:
            executor_class = concurrent.futures.ThreadPoolExecutor
            worker = 1
        elif parallel_processing[0] == "process":
            executor_class = concurrent.futures.ProcessPoolExecutor
            worker = parallel_processing[1]
            raise ValueError('Not supported. Please use thread configuration')
        else:
            executor_class = concurrent.futures.ThreadPoolExecutor
            worker = parallel_processing[1]

        with executor_class(max_workers=worker) as executor:
            executor.map(lambda i: self.calculate_member(i, fitness_fun), enumerate(self.members))

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
