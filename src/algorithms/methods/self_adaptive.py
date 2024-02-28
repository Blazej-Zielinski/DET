import copy
import random
import numpy as np

from diffEvoLib.models.member import Member
from src.models.population import Population
from src.enums.optimization import OptimizationType
from src.enums.strategies import Strategies


def sa_mutation_rand_1(member1: Member, member2: Member, member3: Member, f) -> Member:
    """
        Formula: v_ij = x_r1 + F(x_r2 - x_r3)
    """
    new_member = copy.deepcopy(member1)
    new_member.chromosomes = member1.chromosomes + f * (member2.chromosomes - member3.chromosomes)
    return new_member


def sa_mutation_best_1(best_member: Member, member1: Member, member2: Member, f) -> Member:
    """
        Formula: v_ij = x_best + F(x_r1 - x_r2)
    """
    new_member = copy.deepcopy(best_member)
    new_member.chromosomes = best_member.chromosomes + f * (member1.chromosomes - member2.chromosomes)
    return new_member


def sa_mutation_curr_to_best_1(base_member: Member, best_member: Member, member1: Member, member2: Member, f) -> Member:
    """
        Formula: v_ij = x_base + F(x_best - x_base) + F(x_r1 - x_r2)
    """
    new_member = copy.deepcopy(base_member)
    new_member.chromosomes = (base_member.chromosomes + f * (best_member.chromosomes - base_member.chromosomes) +
                              f * (member1.chromosomes - member2.chromosomes))
    return new_member


def sa_mutation_best_2(best_member: Member, member1: Member, member2: Member, member3: Member, member4: Member,
                       f) -> Member:
    """
        Formula: v_ij = x_best + F(x_r1 - x_r2) + F(x_r3 - x_r4)
    """
    new_member = copy.deepcopy(best_member)
    new_member.chromosomes = (best_member.chromosomes + f * (member1.chromosomes - member2.chromosomes) +
                              f * (member3.chromosomes - member4.chromosomes))
    return new_member


def sa_mutation_rand_2(member1: Member, member2: Member, member3: Member, member4: Member, member5: Member, f):
    """
        Formula: v_ij = x_best + F(x_r1 - x_r2) + F(x_r3 - x_r4)
    """
    new_member = copy.deepcopy(member1)
    new_member.chromosomes = (member1.chromosomes + f * (member2.chromosomes - member3.chromosomes) +
                              f * (member4.chromosomes - member5.chromosomes))
    return new_member


def sa_mutation(population: Population, f, mut_strategy, optimization) -> Population:
    new_members = []
    is_descending = False if optimization == OptimizationType.MINIMIZATION else True

    if mut_strategy == Strategies.RAND_1:
        for _ in range(population.size):
            selected_members = random.sample(population.members.tolist(), 3)
            new_member = sa_mutation_rand_1(selected_members[0], selected_members[1], selected_members[2], f)
            new_members.append(new_member)

    elif mut_strategy == Strategies.BEST_1:
        best_member = population.get_best_members(1)
        for _ in range(population.size):
            selected_members = random.sample(population.members.tolist(), 2)
            new_member = sa_mutation_best_1(best_member, selected_members[0], selected_members[1], f)
            new_members.append(new_member)

    elif mut_strategy == Strategies.CURRENT_TO_BEST_1:
        pop_members_list = population.members.tolist()
        best_member = population.get_best_members(1)
        for i in range(population.size):
            indexes = list(range(i)) + list(range(i + 1, population.size))
            selected_idx = random.sample(indexes, 2)
            new_member = sa_mutation_curr_to_best_1(pop_members_list[i], best_member, pop_members_list[selected_idx[0]],
                                                    pop_members_list[selected_idx[1]], f)
            new_members.append(new_member)

    elif mut_strategy == Strategies.BEST_2:
        best_member = population.get_best_members(1)
        for _ in range(population.size):
            selected_members = random.sample(population.members.tolist(), 4)
            new_member = sa_mutation_best_2(best_member, selected_members[0], selected_members[1], selected_members[3],
                                            selected_members[4], f)
            new_members.append(new_member)

    elif mut_strategy == Strategies.RAND_2:
        for _ in range(population.size):
            selected_members = random.sample(population.members.tolist(), 5)
            new_member = sa_mutation_rand_2(selected_members[0], selected_members[1], selected_members[2],
                                            selected_members[3], selected_members[4], f)
            new_members.append(new_member)

    new_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population
