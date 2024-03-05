import copy
import random
import numpy as np
from math import floor

from src.models.member import Member
from src.models.population import Population

from src.enums.strategies import mutation_curr_to_best_1
from src.algorithms.initializers import draw_cauchy_dist_within_bounds, draw_norm_dist_within_bounds
from src.enums.optimization import OptimizationType


# TODO
# Fi is connected with xi, so every x has its own F
# also F is regenerate in each generation and it is also adapted

# Move strategies from SaDE to somewhere else so they could be used in different algos

# Initializing archive as empty
# In each generation the parent solutions, which fail to success into next gen are added to the archive
# If the size of archive exceeds threshold some of the solutions are randomly removed


def jade_mutation(population: Population, archive: list[Member], mutation_factors: np.ndarray[float], p_best: float):
    pop_members_array = population.members
    pop_archive_members_array = np.concatenate((pop_members_array, np.array(archive)))

    p_best_members = population.get_best_members(floor(p_best * population.size))

    new_members = []
    for (i, base_member) in enumerate(pop_members_array):
        best_member = random.choice(p_best_members)

        indices = list(range(population.size))
        indices.remove(i)
        selected_idx = random.choice(indices)
        x_r1 = pop_members_array[selected_idx]

        indices = list(range(len(pop_archive_members_array)))
        indices.remove(i)
        indices.remove(selected_idx)
        selected_idx = random.choice(indices)
        x_r2 = pop_archive_members_array[selected_idx]

        new_member = mutation_curr_to_best_1(base_member, best_member, [x_r1, x_r2], mutation_factors[i])
        new_members.append(new_member)

    new_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)

    return new_population


def jade_selection(origin_population: Population, modified_population: Population, archive: list[Member],
                   mutation_factors: np.ndarray[float], crossover_rates: np.ndarray[float],
                   success_mutation_factors: list[float], success_crossover_rates: list[float]):
    if origin_population.size != modified_population.size != len(mutation_factors) != len(crossover_rates):
        print("Selection: populations have different sizes")
        return None

    if origin_population.optimization != modified_population.optimization:
        print("Selection: populations have different optimization types")
        return None

    optimization = origin_population.optimization
    new_members = []
    better_count = 0
    for i in range(origin_population.size):
        if optimization == OptimizationType.MINIMIZATION:
            if origin_population.members[i] <= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))
                archive.append(copy.deepcopy(origin_population.members[i]))
                success_mutation_factors.append(mutation_factors[i])
                success_crossover_rates.append(crossover_rates[i])
                better_count += 1

        elif optimization == OptimizationType.MAXIMIZATION:
            if origin_population.members[i] >= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))
                archive.append(copy.deepcopy(origin_population.members[i]))
                success_mutation_factors.append(mutation_factors[i])
                success_crossover_rates.append(crossover_rates[i])
                better_count += 1

    new_population = Population(
        interval=origin_population.interval,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)

    return new_population, archive, success_mutation_factors, success_crossover_rates, better_count


def jade_adapt_mutation_factors(config, success_mutation_factors: list[float]) -> np.ndarray[float] | None:
    if len(success_mutation_factors) == 0:
        return None

    success_factor_mean = sum(success_mutation_factors) / len(success_mutation_factors)
    new_mutation_factors_mean = (1 - config.jade_c) * config.mutation_factor + config.jade_c * success_factor_mean
    config.set_mutation_factor_mean(new_mutation_factors_mean)
    return draw_cauchy_dist_within_bounds(config.mutation_factor_mean, config.mutation_factor_std,
                                          config.population_size, config.mutation_factor_low,
                                          config.mutation_factor_high), []


def jade_adapt_crossover_rates(config, success_crossover_rates: list[float]) -> np.ndarray[float] | None:
    if len(success_crossover_rates) == 0:
        return None

    success_crossover_mean = lehmer_mean(success_crossover_rates)
    new_crossover_rate_mean = (1 - config.jade_c) * config.crossover_rate_mean + config.jade_c * success_crossover_mean
    config.set_crossover_rate_mean(new_crossover_rate_mean)
    return draw_norm_dist_within_bounds(config.crossover_rate_mean, config.crossover_rate_std,
                                        config.population_size, config.crossover_rate_low,
                                        config.crossover_rate_high), []


def lehmer_mean(numbers: list[float]) -> float | None:
    if not numbers or len(numbers) == 0:
        return None

    numbers_array = np.array(numbers)

    numerator = np.sum(numbers_array ** 2)
    denominator = np.sum(numbers_array)

    if denominator == 0:
        return None

    return numerator / denominator


def jade_reduce_archive(pop_size: int, archive: list[Member]) -> list[Member]:
    if len(archive) <= pop_size:
        return archive

    reduce_num = len(archive) - pop_size
    for _ in range(reduce_num):
        idx = random.randrange(len(archive))
        archive.pop(idx)

    return archive


