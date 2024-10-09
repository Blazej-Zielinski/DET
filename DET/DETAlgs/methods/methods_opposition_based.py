from random import random
import numpy as np

from DET.models.member import Member
from DET.models.population import Population


def calculate_central_points(population: Population) -> np.ndarray[float]:
    all_chromosomes = np.array(
        [[chromosome.real_value for chromosome in member.chromosomes] for member in population.members])

    min_values = np.min(all_chromosomes, axis=0)
    max_values = np.max(all_chromosomes, axis=0)

    return min_values + max_values


def calculate_opposite_pop(population: Population, is_initial_pop: bool) -> Population:
    central_points = np.full((population.arg_num,), sum(population.interval)) if is_initial_pop else (
        calculate_central_points(population))

    opposite_members = []
    for member in population.members:
        opposite_member = Member(member.interval, member.args_num)
        for i in range(member.args_num):
            opposite_member.chromosomes[i] = (member.chromosomes[i] - central_points[i]) * -1

        opposite_members.append(opposite_member)

    opposite_population = Population(
        lb=population.lb,
        ub=population.ub,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    opposite_population.members = np.array(opposite_members)

    return opposite_population


def opp_based_keep_best_individuals(population: Population, func, is_initial_pop: bool = False):
    opposite_population = calculate_opposite_pop(population, is_initial_pop)
    opposite_population.update_fitness_values(func)

    pops = np.concatenate((population.members, opposite_population.members))
    sorted_pops_indices = np.argsort([member.fitness_value for member in pops])
    sorted_pops = pops[sorted_pops_indices]
    population.members = sorted_pops[:population.size]


def opp_based_generation_jumping(population: Population, jumping_rate: float, func) -> bool:
    if random() < jumping_rate:
        opp_based_keep_best_individuals(population, func)
        return True
    return False
