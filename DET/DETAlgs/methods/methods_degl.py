import random
import numpy as np

from src.models.population import Population
from src.models.member import Member
from diffEvoLib.models.enums.mutation import mutation_curr_to_best_1


def generate_local_donor(base_member: Member, neighborhood: Population, f: float) -> Member:
    best_local = neighborhood.get_best_members(1)[0]
    selected_members = random.sample(neighborhood.members.tolist(), 2)
    local_donor = mutation_curr_to_best_1(base_member, best_local, selected_members, f)
    return local_donor


def combine_donors(global_donor: Member, local_donor: Member, weight: float) -> Member:
    return global_donor * weight + local_donor * (1 - weight)


def degl_mutation(population: Population, radius: int, f: float, weight: float) -> Population:
    pop_members_list = population.members.tolist()
    best_global = population.get_best_members(1)[0]

    new_members = []
    population_indices = list(range(population.size))
    topology_indices = population_indices[-radius:] + population_indices + population_indices[:radius]
    for i, member in enumerate(population.members):
        # Generate local donor
        local_indices = topology_indices[i:i + 2 * radius + 1]
        local_indices.remove(i)
        neighborhood_members = [pop_members_list[i] for i in local_indices]
        neighborhood = Population(
            interval=population.interval,
            arg_num=population.arg_num,
            size=2 * radius,
            optimization=population.optimization
        )
        neighborhood.members = np.array(neighborhood_members)
        local_donor = generate_local_donor(member, neighborhood, f)

        # Generating global donor
        global_indices = list(range(population.size))
        global_indices.remove(i)
        selected_indices = random.sample(global_indices, 2)
        global_members = [pop_members_list[i] for i in selected_indices]
        global_donor = mutation_curr_to_best_1(member, best_global, global_members, f)

        # Combining donors into new member
        new_member = combine_donors(global_donor, local_donor, weight)
        new_members.append(new_member)

    new_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)

    return new_population


def degl_adapt_weight(current_generation: int, max_generation: int) -> float:
    return current_generation / max_generation
