import copy
import random
import numpy as np

from src.models.member import Member
from src.models.population import Population
from src.enums.strategies import StrategiesEnum
from src.models.strategy import Strategy
from src.enums.optimization import OptimizationType
from src.algorithms.methods.default_de import binomial_crossing_ind


def sa_mutation_rand_1(members: list[Member], f: float) -> Member:
    """
        Formula: v_ij = x_r1 + F(x_r2 - x_r3)
    """
    new_member = copy.deepcopy(members[0])
    new_member.chromosomes = members[0].chromosomes + (members[1].chromosomes - members[2].chromosomes) * f
    return new_member


def sa_mutation_best_1(best_member: Member, members: list[Member], f: float) -> Member:
    """
        Formula: v_ij = x_best + F(x_r1 - x_r2)
    """
    new_member = copy.deepcopy(best_member)
    new_member.chromosomes = best_member.chromosomes + (members[0].chromosomes - members[1].chromosomes) * f
    return new_member


def sa_mutation_curr_to_best_1(base_member: Member, best_member: Member, members: list[Member], f: float) -> Member:
    """
        Formula: v_ij = x_base + F(x_best - x_base) + F(x_r1 - x_r2)
    """
    new_member = copy.deepcopy(base_member)
    new_member.chromosomes = (base_member.chromosomes + (best_member.chromosomes - base_member.chromosomes) * f +
                              (members[0].chromosomes - members[1].chromosomes) * f)
    return new_member


def sa_mutation_best_2(best_member: Member, members: list[Member], f: float) -> Member:
    """
        Formula: v_ij = x_best + F(x_r1 - x_r2) + F(x_r3 - x_r4)
    """
    new_member = copy.deepcopy(best_member)
    new_member.chromosomes = (best_member.chromosomes + (members[0].chromosomes - members[1].chromosomes) * f +
                              (members[2].chromosomes - members[3].chromosomes) * f)
    return new_member


def sa_mutation_rand_2(members: list[Member], f: float) -> Member:
    """
        Formula: v_ij = x_best + F(x_r1 - x_r2) + F(x_r3 - x_r4)
    """
    new_member = copy.deepcopy(members[0])
    new_member.chromosomes = (members[0].chromosomes + (members[1].chromosomes - members[2].chromosomes) * f +
                              (members[3].chromosomes - members[4].chromosomes) * f)
    return new_member


def sa_get_mutation_strategy(mutation_strategies: list[Strategy], member_strategy_indicator: float) -> Strategy:
    if len(mutation_strategies) != 2:
        raise ValueError("Wrong number of strategies")

    if member_strategy_indicator <= mutation_strategies[0].probability:
        return mutation_strategies[0]
    else:
        return mutation_strategies[1]


def get_strategy_function(strategy_type: StrategiesEnum):
    return {
        StrategiesEnum.RAND_1: (lambda base, best, members, f: sa_mutation_rand_1(members, f)),
        StrategiesEnum.RAND_2: (lambda base, best, members, f: sa_mutation_rand_2(members, f)),
        StrategiesEnum.BEST_1: (lambda base, best, members, f: sa_mutation_best_1(best, members, f)),
        StrategiesEnum.BEST_2: (lambda base, best, members, f: sa_mutation_best_2(best, members, f)),
        StrategiesEnum.CURRENT_TO_BEST_1: (
            lambda base, best, members, f: sa_mutation_curr_to_best_1(base, best, members, f)),
    }.get(strategy_type, lambda: None)


def sa_mutation(population: Population, mutation_factors: np.ndarray[float], mutation_strategies: list[Strategy],
                mutation_strategy_indicators: np.ndarray[float]) -> tuple[Population, list[Strategy]]:
    if population.size != len(mutation_factors) != len(mutation_strategy_indicators):
        raise ValueError("Different sizes of parameters!")

    new_members = []
    drawn_strategy = []

    pop_members_list = population.members.tolist()
    best_member = population.get_best_members(1)[0]

    for (idx, indicator) in enumerate(mutation_strategy_indicators):
        member_strategy = sa_get_mutation_strategy(mutation_strategies, indicator)
        drawn_strategy.append(member_strategy)
        strategy_function = get_strategy_function(member_strategy.strategy)

        indices = list(range(idx)) + list(range(idx + 1, population.size))
        selected_indices = random.sample(indices, 5)
        additional_members = [pop_members_list[i] for i in selected_indices]

        new_member = strategy_function(pop_members_list[idx], best_member, additional_members, mutation_factors[idx])
        new_members.append(new_member)

    new_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)

    return new_population, drawn_strategy


def sa_binomial_crossing(origin_population: Population, mutated_population: Population,
                         crossover_rates: np.ndarray[float]) -> Population:
    if origin_population.size != mutated_population.size != len(crossover_rates):
        raise ValueError("Binomial_crossing: populations have different sizes")

    new_members = []
    for i, cr in enumerate(crossover_rates):
        new_member = binomial_crossing_ind(origin_population.members[i], mutated_population.members[i], cr)
        new_members.append(new_member)

    new_population = Population(
        interval=origin_population.interval,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)

    return new_population


def sa_selection(origin_population: Population, modified_population: Population, member_strategies: list[Strategy],
                 crossover_rates: np.ndarray[float], crossover_success_rates: list[float]):
    if origin_population.size != modified_population.size != len(member_strategies):
        print("Selection: populations have different sizes")
        return None

    if origin_population.optimization != modified_population.optimization:
        print("Selection: populations have different optimization types")
        return None

    optimization = origin_population.optimization
    new_members = []
    better_count = 0
    for i, strategy in enumerate(member_strategies):
        if optimization == OptimizationType.MINIMIZATION:
            if origin_population.members[i] <= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
                strategy.update_nf()
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))
                strategy.update_ns()
                crossover_success_rates.append(crossover_rates[i])
                better_count += 1
        elif optimization == OptimizationType.MAXIMIZATION:
            if origin_population.members[i] >= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
                strategy.update_nf()
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))
                strategy.update_ns()
                crossover_success_rates.append(crossover_rates[i])
                better_count += 1

    new_population = Population(
        interval=origin_population.interval,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)

    return new_population, better_count


def sa_adapt_probabilities(strategy1: Strategy, strategy2: Strategy):
    ns1 = strategy1.ns
    nf1 = strategy1.nf

    ns2 = strategy2.ns
    nf2 = strategy2.nf

    numerator = ns1 * (ns2 + nf2)
    denominator = ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2)

    p1 = numerator / denominator if denominator != 0 else 0
    p2 = 1 - p1

    strategy1.set_probability(p1)
    strategy2.set_probability(p2)

    strategy1.reset_nf_ns()
    strategy2.reset_nf_ns()


def sa_adapt_crossover_rates(config, crossover_success_rates: list[float]):
    new_cross_mean = sum(crossover_success_rates) / len(crossover_success_rates) if len(crossover_success_rates) != 0 else 0
    config.set_crossover_mean(new_cross_mean)
    return np.random.normal(loc=config.crossover_rate_mean, scale=config.crossover_rate_std,
                            size=config.population_size)
