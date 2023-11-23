import numpy as np


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
