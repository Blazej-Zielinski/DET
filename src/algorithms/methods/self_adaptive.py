import copy
import random
from typing import Tuple, List

import numpy as np

from diffEvoLib.models.member import Member
from src.models.population import Population
from src.enums.strategies import StrategiesEnum
from src.models.strategy import Strategy
from src.enums.optimization import OptimizationType


def sa_mutation_rand_1(members: list[Member], f: float) -> Member:
    """
        Formula: v_ij = x_r1 + F(x_r2 - x_r3)
    """
    new_member = copy.deepcopy(members[0])
    new_member.chromosomes = members[0].chromosomes + f * (members[1].chromosomes - members[2].chromosomes)
    return new_member


def sa_mutation_best_1(best_member: Member, members: list[Member], f: float) -> Member:
    """
        Formula: v_ij = x_best + F(x_r1 - x_r2)
    """
    new_member = copy.deepcopy(best_member)
    new_member.chromosomes = best_member.chromosomes + f * (members[0].chromosomes - members[1].chromosomes)
    return new_member


def sa_mutation_curr_to_best_1(base_member: Member, best_member: Member, members: list[Member], f: float) -> Member:
    """
        Formula: v_ij = x_base + F(x_best - x_base) + F(x_r1 - x_r2)
    """
    new_member = copy.deepcopy(base_member)
    new_member.chromosomes = (base_member.chromosomes + f * (best_member.chromosomes - base_member.chromosomes) +
                              f * (members[0].chromosomes - members[1].chromosomes))
    return new_member


def sa_mutation_best_2(best_member: Member, members: list[Member], f: float) -> Member:
    """
        Formula: v_ij = x_best + F(x_r1 - x_r2) + F(x_r3 - x_r4)
    """
    new_member = copy.deepcopy(best_member)
    new_member.chromosomes = (best_member.chromosomes + f * (members[0].chromosomes - members[1].chromosomes) +
                              f * (members[2].chromosomes - members[3].chromosomes))
    return new_member


def sa_mutation_rand_2(members: list[Member], f: float) -> Member:
    """
        Formula: v_ij = x_best + F(x_r1 - x_r2) + F(x_r3 - x_r4)
    """
    new_member = copy.deepcopy(members[0])
    new_member.chromosomes = (members[0].chromosomes + f * (members[1].chromosomes - members[2].chromosomes) +
                              f * (members[3].chromosomes - members[4].chromosomes))
    return new_member


def sa_get_mutation_strategy(mutation_strategies: list[Strategy], member_strategy_indicator: float) -> Strategy:
    len_strategies = len(mutation_strategies)

    if len_strategies < 1 or len_strategies > 5:
        raise ValueError("Wrong number of strategies")

    if len_strategies == 1:
        return mutation_strategies[0]

    elif len_strategies == 2:
        if member_strategy_indicator <= mutation_strategies[0].probability:
            return mutation_strategies[0]
        else:
            return mutation_strategies[1]

    else:
        for strategy in mutation_strategies[:-1]:
            if member_strategy_indicator <= strategy.probability:
                return strategy

        return mutation_strategies[-1]


def get_strategy_function(strategy_type: StrategiesEnum):
    return {
        StrategiesEnum.RAND_1: (lambda base, best, members, f: sa_mutation_rand_1(members, f)),
        StrategiesEnum.RAND_2: (lambda base, best, members, f: sa_mutation_rand_2(members, f)),
        StrategiesEnum.BEST_1: (lambda base, best, members, f: sa_mutation_best_1(best, members, f)),
        StrategiesEnum.BEST_2: (lambda base, best, members, f: sa_mutation_best_2(best, members, f)),
        StrategiesEnum.CURRENT_TO_BEST_1: (
            lambda base, best, members, f: sa_mutation_curr_to_best_1(base, best, members, f)),
    }.get(strategy_type, lambda: None)


def sa_mutation(population: Population, f: float, mutation_strategies: list[Strategy]) -> \
        tuple[Population, list[Strategy]]:
    new_members = []
    drew_strategy = []

    pop_members_list = population.members.tolist()
    best_member = population.get_best_members(1)

    member_strategy_indicators = np.random.uniform(low=0, high=1, size=population.size)
    for (idx, indicator) in enumerate(member_strategy_indicators):
        member_strategy = sa_get_mutation_strategy(mutation_strategies, indicator)
        drew_strategy.append(member_strategy)
        strategy_function = get_strategy_function(member_strategy.strategy)

        indices = list(range(idx)) + list(range(idx + 1, population.size))
        selected_indices = random.sample(indices, 5)
        auxiliary_members = [pop_members_list[i] for i in selected_indices]

        new_member = strategy_function(pop_members_list[idx], best_member, auxiliary_members, f)
        new_members.append(new_member)

    new_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)

    return new_population, drew_strategy


def sa_selection(origin_population: Population, modified_population: Population, member_strategies: list[Strategy]):
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
                better_count += 1
        elif optimization == OptimizationType.MAXIMIZATION:
            if origin_population.members[i] >= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
                strategy.update_nf()
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))
                strategy.update_ns()
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

    p1 = (ns1 * (ns2 + nf2)) / (ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2))
    p2 = 1 - p1

    strategy1.set_probability(p1)
    strategy2.set_probability(p2)
