import random
import numpy as np
from collections import deque

from src.models.population import Population
from src.models.member import Member
from src.enums.strategies import mutation_curr_to_best_1


def degl_mutation_local(base_member: Member, neighborhood: Population, f: float) -> Member:
    best_local = neighborhood.get_best_members(1)[0]
    selected_members = random.sample(neighborhood.members.tolist(), 2)
    local_donor = mutation_curr_to_best_1(base_member, best_local, selected_members, f)
    return local_donor


def degl_combine_donors(global_donor: Member, local_donor: Member, w: float) -> Member:
    for i in range(global_donor.args_num):
        global_donor.chromosomes[i].real_value += w * local_donor.chromosomes[i].real_value

    return global_donor


def degl_mutation(population: Population, k: int, f: float, w: float) -> Population:
    pop_members_list = population.members.tolist()
    best_global = population.get_best_members(1)[0]

    new_members = []
    neighborhood_indices = list(range(population.size))
    neighborhood_indices = neighborhood_indices[-k:] + neighborhood_indices + neighborhood_indices[:k]
    for i, member in enumerate(population.members):
        # Generating local donor
        start = population.size + i - k
        local_indices = neighborhood_indices[i:i + 2 * k + 1]
        local_indices.remove(i)
        neighborhood_members = [pop_members_list[i] for i in local_indices]
        neighborhood = Population(
            interval=population.interval,
            arg_num=population.arg_num,
            size=2 * k,
            optimization=population.optimization
        )
        neighborhood.members = np.array(neighborhood_members)

        local_donor = degl_mutation_local(member, neighborhood, f)

        # Generating global donor
        global_indices = list(range(i)) + list(range(i + 1, population.size))
        selected_indices = random.sample(global_indices, 2)
        global_members = [pop_members_list[i] for i in selected_indices]
        global_donor = mutation_curr_to_best_1(member, best_global, global_members, f)

        # Combining donors into new member
        new_member = degl_combine_donors(global_donor, local_donor, w)
        new_members.append(new_member)

    new_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)

    return new_population
