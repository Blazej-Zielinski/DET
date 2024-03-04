import copy
import random
import numpy as np

from src.models.member import Member

from src.enums.strategies import mutation_curr_to_best_1
from src.algorithms.initializers import draw_cauchy_dist_within_bounds, draw_norm_dist_within_bounds


# TODO
# Fi is connected with xi, so every x has its own F
# also F is regenerate in each generation and it is also adapted

# Move strategies from SaDE to somewhere else so they could be used in different algos

# Initializing archive as empty
# In each generation the parent solutions, which fail to success into next gen are added to the archive
# If the size of archive exceeds threshold some of the solutions are randomly removed

def jade_mutation():
    pass


def jade_adapt_mutation_factors(config, success_mutation_factors: list[float]) -> np.ndarray[float] | None:
    if len(success_mutation_factors) == 0:
        return None

    success_factor_mean = sum(success_mutation_factors) / len(success_mutation_factors)
    new_mutation_factors_mean = (1 - config.jade_c) * config.mutation_factor + config.jade_c * success_factor_mean
    config.set_mutation_factors(new_mutation_factors_mean)
    return draw_cauchy_dist_within_bounds(config.mutation_factor_mean, config.mutation_factor_std,
                                          config.population_size, config.mutation_factor_low,
                                          config.mutation_factor_high)


def jade_adapt_crossover_rates(config, success_crossover_rates: list[float]) -> np.ndarray[float] | None:
    if len(success_crossover_rates) == 0:
        return None

    success_crossover_mean = lehmer_mean(success_crossover_rates)
    new_crossover_rates_mean = (1 - config.jade_c) * config.crossover_rate_mean + config.jade_c * success_crossover_mean
    config.set_crossover_rates(new_crossover_rates_mean)
    return draw_norm_dist_within_bounds(config.crossover_rate_mean, config.crossover_rate_std,
                                        config.population_size, config.crossover_rate_low,
                                        config.crossover_rate_high)


def lehmer_mean(numbers: list[float]) -> float | None:
    if len(numbers) == 0:
        return None

    numerator = np.sum(np.array(numbers) ** 2)
    denominator = sum(numbers)

    return numerator / denominator
