import numpy as np
from src.models.strategy import Strategy


def nm_initialize(config):
    delta_f = 0.2
    delta_cr = 0.1
    sp = 50
    flags = np.zeros(config.population_size)
    f_arr = np.random.uniform(size=config.population_size, low=0, high=2)
    cr_arr = np.random.uniform(size=config.population_size, low=0, high=1)
    f_set = set()
    cr_set = set()

    return delta_f, delta_cr, sp, flags, f_arr, cr_arr, f_set, cr_set


def pbx_initialize():
    f_m = 0.5
    cr_m = 0.6

    return f_m, cr_m


def adaptive_params_initialize(config):
    f_arr = np.random.uniform(size=config.population_size)
    cr_arr = np.random.uniform(size=config.population_size)
    prob_f = 0.1
    prob_cr = 0.1

    return f_arr, cr_arr, prob_f, prob_cr


def sa_initialize(config) -> tuple:
    if len(config.mutation_strategies) != 2:
        raise Exception("Wrong number of mutation strategies!")

    mutation_factors = draw_norm_dist_within_bounds(config.mutation_factor_mean, config.mutation_factor_std,
                                                    config.population_size, config.mutation_factor_low,
                                                    config.mutation_factor_high)
    crossover_rates = np.random.normal(loc=config.crossover_rate_mean, scale=config.crossover_rate_std,
                                       size=config.population_size)
    crossover_success_rates = []
    mutation_strategies = [Strategy(stg, 1 / (len(config.mutation_strategies))) for stg in config.mutation_strategies]
    mutation_strategy_indicators = np.random.uniform(low=0, high=1, size=config.population_size)

    return mutation_factors, mutation_strategies, crossover_rates, mutation_strategy_indicators, crossover_success_rates


def jade_initialize(config) -> tuple:
    mutation_factors = draw_cauchy_dist_within_bounds(config.mutation_factor_mean, config.mutation_factor_std,
                                                      config.population_size, config.mutation_factor_low,
                                                      config.mutation_factor_high)

    crossover_rates = draw_norm_dist_within_bounds(config.crossover_rate_mean, config.crossover_rate_std,
                                                   config.population_size, config.crossover_rate_low,
                                                   config.crossover_rate_high)
    success_mutation_factors = []
    success_crossover_rates = []
    archive = []

    return mutation_factors, crossover_rates, success_mutation_factors, success_crossover_rates, archive


def opposition_based_de_initialize(config) -> tuple:
    pass


def draw_norm_dist_within_bounds(mean: float, std: float, arr_size: int, low: float, high: float) -> np.ndarray:
    """
    Draw numbers from normal distributions within bounds (low, high].
    """
    values = np.clip(np.random.normal(loc=mean, scale=std, size=arr_size), low, high)

    while len(values) < arr_size:
        new_values = np.clip(np.random.normal(loc=mean, scale=std, size=arr_size - len(values)), low, high)
        values = np.concatenate((values, new_values))

    return values[:arr_size]


# def draw_cauchy_dist_within_bounds(mean: float, std: float, arr_size: int, low: int, high: int) -> np.ndarray:
#     """
#         Draw numbers from Cauchy distribution within bounds (low,high].
#     """
#     values = mean + std * np.random.standard_cauchy(size=arr_size)
#
#     values = np.where(values >= high, high, values)
#     values = np.where(values <= low, draw_cauchy_dist_within_bounds(mean, std, arr_size, low, high), values)
#
#     return values


def draw_cauchy_dist_within_bounds(mean: float, std: float, arr_size: int, low: int, high: int) -> np.ndarray:
    values = np.empty(arr_size)
    i = 0

    while i < arr_size:
        value = mean + std * np.random.standard_cauchy()

        if value >= 1:
            values[i] = 1
            i += 1
        elif 0 < value < 1:
            values[i] = value
            i += 1

    return values
