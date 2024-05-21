import numpy as np
from math import floor
import random
import copy

from diffEvoLib.models.population import Population
from diffEvoLib.models.enums.optimization import OptimizationType


def aade_selection(origin_population: Population, modified_population: Population, mutation_factors: list[float],
                   crossover_rates: list[float], f: float, cr: float) -> Population:
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
                mutation_factors.append(
                    f - (modified_population.members[i].fitness_value - origin_population.members[i].fitness_value) /
                    modified_population.members[i].fitness_value * random.random())

                crossover_rates.append(
                    cr - (modified_population.members[i].fitness_value - origin_population.members[i].fitness_value) /
                    modified_population.members[i].fitness_value * random.random())
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))
        elif optimization == OptimizationType.MAXIMIZATION:
            if origin_population.members[i] >= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
                mutation_factors.append(
                    f - (modified_population.members[i].fitness_value - origin_population.members[i].fitness_value) /
                    modified_population.members[i].fitness_value * random.random())

                crossover_rates.append(
                    cr - (modified_population.members[i].fitness_value - origin_population.members[i].fitness_value) /
                    modified_population.members[i].fitness_value * random.random())
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


def aade_adapat_parameters(mutation_factors: list[float], crossover_rates: list[float]) -> tuple[float, float]:
    min_f = min(mutation_factors)
    max_f = max(mutation_factors)
    tmp_f = (max_f - min_f) / max_f
    mutation_factors.clear()

    min_cr = min(crossover_rates)
    max_cr = max(crossover_rates)
    tmp_cr = (max_cr - min_cr) / max_cr
    crossover_rates.clear()

    return random.random() * tmp_f, random.random() * tmp_cr
