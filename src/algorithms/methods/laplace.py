import random
import numpy as np
import copy

from src.models.member import Member
from src.models.population import Population


# working poor
def laplace_mutation_ind(base_member: Member, member1: Member):
    """
        Formula: v_ij = x_r1 + L * |x_r1 - x_r2|
    """
    new_member = copy.deepcopy(base_member)

    absolute_difference = abs(base_member.chromosomes - member1.chromosomes)
    new_member.chromosomes = base_member.chromosomes + absolute_difference * np.random.laplace(0, 1)
    return new_member


# working quite well (Schema 5)
def laplace_mutation_ind2(best_member: Member, member1: Member, member2: Member):
    """
        Formula: v_ij = x_r1 + L * |x_best - x_r2|
    """
    new_member = copy.deepcopy(best_member)

    absolute_difference = abs(best_member.chromosomes - member2.chromosomes)
    new_member.chromosomes = member1.chromosomes + absolute_difference * np.random.laplace(0, 1)
    return new_member


# working medium well (Schema 4)
def laplace_mutation_ind3(base_member: Member, member1: Member, member2: Member, f=0.5):
    """
        Formula: v_ij = x_r1 + L * |x_best - x_r2|
    """
    new_member = copy.deepcopy(base_member)

    if np.random.uniform() > 0.2:
        absolute_difference = abs(base_member.chromosomes - member1.chromosomes)
        new_member.chromosomes = base_member.chromosomes + absolute_difference * np.random.laplace(0, 1)
    else:
        new_member.chromosomes = base_member.chromosomes + (member1.chromosomes - member2.chromosomes) * f

    return new_member


def laplace_mutation(population: Population):
    new_members = []
    for _ in range(population.size):
        best_member = population.get_best_members(1)[0]
        selected_members = random.sample(population.members.tolist(), 3)
        # new_member = laplace_mutation_ind(selected_members[0], selected_members[1])
        new_member = laplace_mutation_ind2(best_member, selected_members[0], selected_members[1])
        #new_member = laplace_mutation_ind3(selected_members[0], selected_members[1], selected_members[2])

        new_members.append(new_member)

    new_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population
