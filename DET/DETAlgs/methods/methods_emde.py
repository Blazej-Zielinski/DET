import random
import numpy as np
import copy

from diffEvoLib.models.member import Member
from diffEvoLib.models.population import Population


def em_mutation_ind(best_member: Member, better_member: Member, worst_member: Member):
    """
        Formula: v_i = x_c + F1(x_best - x_better) + F2(x_best - x_worst) + F3(x_better - x_worst)
    """
    new_member = copy.deepcopy(best_member)

    fs = np.random.uniform(size=3)
    w1, w2, w3 = get_weights()

    member_c = best_member.chromosomes * w1 + better_member.chromosomes * w2 + worst_member.chromosomes * w3
    f1_component = (best_member.chromosomes - better_member.chromosomes) * fs[0]
    f2_component = (best_member.chromosomes - worst_member.chromosomes) * fs[1]
    f3_component = (better_member.chromosomes - worst_member.chromosomes) * fs[2]
    new_member.chromosomes = member_c + f1_component + f2_component + f3_component

    return new_member


def em_mutation(population: Population):
    new_members = []
    for _ in range(population.size):
        selected_members = random.sample(population.members.tolist(), 3)
        new_member = em_mutation_ind(selected_members[0], selected_members[1], selected_members[2])
        new_members.append(new_member)

    new_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population


def get_weights():
    p1 = 1.0
    p2 = np.random.uniform(low=0.75, high=1.0)
    p3 = np.random.uniform(low=0.5, high=p2)
    p_sum = p1 + p2 + p3

    w1 = p1 / p_sum
    w2 = p2 / p_sum
    w3 = p3 / p_sum

    return w1, w2, w3
