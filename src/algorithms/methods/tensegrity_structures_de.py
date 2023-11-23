import random
import numpy as np
import copy

from src.models.member import Member
from src.models.population import Population


def ts_mutation_ind(best_member: Member, member1: Member, member2: Member, f):
    """
        Formula: v_ij = x_best + F(x_r2 - x_r3)
    """
    new_member = copy.deepcopy(best_member)
    new_member.chromosomes = best_member.chromosomes + (member1.chromosomes - member2.chromosomes) * f
    return new_member


def ts_mutation(population: Population):
    new_members = []
    f = calculate_f()
    for _ in range(population.size):
        sorted_members = population.get_best_members(population.size)
        selected_members = random.sample(sorted_members[1:].tolist(), 2)
        new_member = ts_mutation_ind(sorted_members[0], selected_members[0], selected_members[1], f)

        new_members.append(new_member)

    new_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population


def calculate_f():
    return 2.0 * np.random.uniform() - 1.0


def ts_selection(origin_population: Population, modified_population: Population):
    if origin_population.size != modified_population.size:
        print("Selection: populations have different sizes")
        return None

    if origin_population.optimization != modified_population.optimization:
        print("Selection: populations have different optimization types")
        return None

    all_members = np.concatenate((origin_population.members, modified_population.members))
    sorted_indices = np.argsort([member.fitness_value for member in all_members])
    sorted_members = all_members[sorted_indices]

    new_members = sorted_members[:origin_population.size]

    new_population = Population(
        interval=origin_population.interval,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population
