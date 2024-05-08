import math
from src.models.fuzzy_universe import FuzzyLogicControl
from src.models.population import Population


def fuzzy_compute_function_vector_change(origin_population: Population, next_population: Population) -> float:
    FC = [(n_member.fitness_value - o_member.fitness_value) ** 2 for o_member, n_member in
          zip(origin_population.members, next_population.members)]
    FC = sum(FC) / origin_population.size
    FC = math.sqrt(FC)

    return FC


def fuzzy_compute_parameter_vector_change(origin_population: Population, next_population: Population) -> float:
    PC = [n_member - o_member for o_member, n_member in zip(origin_population.members, next_population.members)]
    PC = [sum([chromosome.real_value**2 for chromosome in member.chromosomes]) for member in PC]
    PC = sum(PC) / origin_population.size
    PC = math.sqrt(PC)

    return PC


def fuzzy_compute_error(vtr: float, best_func_value: float) -> float:
    return vtr - best_func_value


def fuzzy_input_params(origin_population: Population, next_population: Population, vtr: float,
                       best_func_value: float) -> tuple[list[float], list[float]]:
    best_func_value_origin = origin_population.get_best_members(1)[0].fitness_value
    best_func_value_next = next_population.get_best_members(1)[0].fitness_value

    PC = fuzzy_compute_parameter_vector_change(origin_population, next_population)
    FC = fuzzy_compute_function_vector_change(origin_population, next_population)
    e = fuzzy_compute_error(best_func_value_origin, best_func_value_next)

    tmp = math.pow(e, -PC)

    d11 = 1 - (1 + PC) * math.pow(e, -PC)
    d12 = 1 - (1 + FC) * math.pow(e, -FC)

    d21 = 2 * (1 - (1 + PC)) * math.pow(e, -PC)
    d22 = 2 * (1 - (1 + FC)) * math.pow(e, -FC)

    return [d11, d12], [d21, d22]


def fuzzy_adapt_f_cr(FLC: FuzzyLogicControl, origin_population: Population, next_population: Population, vtr: float,
                     best_func_value: float) -> tuple[float, float]:
    f_input, cr_input = fuzzy_input_params(origin_population, next_population, vtr, best_func_value)

    new_f = FLC.f_system.compute_output(*f_input)
    new_cr = FLC.cr_system.compute_output(*cr_input)

    return new_f, new_cr
