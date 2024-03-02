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


def sa_initialize(config):
    if len(config.mutation_strategies) != 2:
        raise Exception("Wrong number of mutation strategies!")

    mutation_factors = initialize_mutation_factors_normal_dist(config.mutation_factor_mean, config.mutation_factor_std,
                                                               config.population_size, config.mutation_factor_low,
                                                               config.mutation_factor_high)
    crossover_rates = np.random.normal(loc=config.crossover_rate_mean, scale=config.crossover_rate_std,
                                       size=config.population_size)
    crossover_success_rates = []
    mutation_strategies = [Strategy(stg, 1 / (len(config.mutation_strategies))) for stg in config.mutation_strategies]
    mutation_strategy_indicators = np.random.uniform(low=0, high=1, size=config.population_size)

    return [mutation_factors, mutation_strategies, crossover_rates, mutation_strategy_indicators,
            crossover_success_rates]


def initialize_mutation_factors_normal_dist(mean: float, std: float, arr_size: int, low: float,
                                            high: float) -> np.ndarray:
    values = np.random.normal(loc=mean, scale=std, size=arr_size)

    while True:
        mask = (low < values) & (values <= high)
        values = values[mask]

        size_vals = len(values)
        if size_vals == arr_size:
            return values

        new_values = np.random.normal(loc=mean, scale=std, size=arr_size - size_vals)
        values = np.concatenate((values, new_values))
