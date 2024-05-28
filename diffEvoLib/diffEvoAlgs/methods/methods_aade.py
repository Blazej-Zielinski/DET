import numpy as np
import random
import copy

from diffEvoLib.diffEvoAlgs.methods.methods_default import mutation_ind, binomial_crossing_ind
from diffEvoLib.models.population import Population
from diffEvoLib.models.enums.optimization import OptimizationType


def aade_mutation(population: Population, mutation_factors: list[list[float, bool]]) -> Population:
    new_members = []
    members = population.members

    for i in range(population.size):
        idxs = np.random.choice(population.size, 3, replace=False)
        selected_members = members[idxs]
        new_member = mutation_ind(selected_members[0], selected_members[1], selected_members[2], mutation_factors[i][0])
        new_members.append(new_member)

    new_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population


def aade_crossing(origin_population: Population, mutated_population: Population,
                  crossover_rates: list[list[float, bool]]) -> Population | None:
    if origin_population.size != mutated_population.size:
        print("Binomial_crossing: populations have different sizes")
        return None

    new_members = []

    for i in range(origin_population.size):
        new_member = binomial_crossing_ind(origin_population.members[i],
                                           mutated_population.members[i],
                                           crossover_rates[i][0])
        new_members.append(new_member)

    new_population = Population(
        interval=origin_population.interval,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population


def aade_selection(origin_population: Population, modified_population: Population,
                   mutation_factors: list[list[float, bool]],
                   crossover_rates: list[list[float, bool]]) -> Population | None:
    if origin_population.size != modified_population.size:
        print("Selection: populations have different sizes")
        return None

    if origin_population.optimization != modified_population.optimization:
        print("Selection: populations have different optimization types")
        return None

    optimization = origin_population.optimization
    new_members = []
    for i in range(origin_population.size):
        if optimization == OptimizationType.MINIMIZATION:
            if origin_population.members[i] <= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
                try:
                    mutation_factors[i] = [mutation_factors[i][0] - (
                            modified_population.members[i].fitness_value - origin_population.members[i].fitness_value) /
                                           modified_population.members[i].fitness_value * random.random(), True]
                except Exception as e:
                    mutation_factors[i] = [0, True]
                try:
                    crossover_rates[i] = [crossover_rates[i][0] - (
                            modified_population.members[i].fitness_value - origin_population.members[i].fitness_value) /
                                          modified_population.members[i].fitness_value * random.random(), True]
                except Exception as e:
                    crossover_rates[i] = [0, True]
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))
        elif optimization == OptimizationType.MAXIMIZATION:
            if origin_population.members[i] >= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
                mutation_factors[i] = [mutation_factors[i][0] - (
                        modified_population.members[i].fitness_value - origin_population.members[i].fitness_value) /
                                       modified_population.members[i].fitness_value * random.random(), True]
                crossover_rates[i] = [crossover_rates[i][0] - (
                        modified_population.members[i].fitness_value - origin_population.members[i].fitness_value) /
                                      modified_population.members[i].fitness_value * random.random(), True]
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))

    new_population = Population(
        interval=origin_population.interval,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population


def aade_adapat_parameters(population: Population, mutation_factors: list[list[float, bool]],
                           crossover_rates: list[list[float, bool]]):
    sorted_members = population.get_best_members(population.size)
    max_fitness_value = sorted_members[-1].fitness_value
    min_fitness_value = sorted_members[0].fitness_value

    for i in range(population.size):
        if mutation_factors[i][1]:
            mutation_factors[i][1] = False
        else:
            mutation_factors[i] = [random.random() * ((max_fitness_value - min_fitness_value) / max_fitness_value),
                                   False]

        if crossover_rates[i][1]:
            crossover_rates[i][1] = False
        else:
            crossover_rates[i] = [random.random() * ((max_fitness_value - min_fitness_value) / max_fitness_value),
                                  False]
