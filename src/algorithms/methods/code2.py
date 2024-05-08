import random
import numpy as np
import copy
import math

from src.models.member import Member
from src.models.member import Chromosome
from src.models.population import Population


def code_constraint_val(chromosome: Chromosome) -> float:
    lower_boundary = chromosome.interval[0]
    upper_boundary = chromosome.interval[1]

    if chromosome.real_value <= lower_boundary:
        return abs(lower_boundary - chromosome.real_value)

    elif upper_boundary <= chromosome.real_value:
        return abs(lower_boundary - chromosome.real_value)

    return 0


def code_check_if_feasible(member: Member) -> tuple[bool, float]:
    constraint_violation = sum([code_constraint_val(chromosome) for chromosome in member.chromosomes])

    if constraint_violation == 0:
        return True, 0
    else:
        return False, constraint_violation


def code_feasibility_objective_func(member: Member, worst_fitness_val: float) -> tuple[bool, float]:
    is_feasible, constraint_violation = code_check_if_feasible(member)

    if is_feasible:
        return True, member.fitness_value
    else:
        return False, worst_fitness_val + constraint_violation


def code_feasibility_rule(members: list[Member], worst_fitness_val: float) -> Member:
    feasibility_1, violation_1 = code_feasibility_objective_func(members[0], worst_fitness_val)
    feasibility_2, violation_2 = code_feasibility_objective_func(members[1], worst_fitness_val)
    feasibility_3, violation_3 = code_feasibility_objective_func(members[2], worst_fitness_val)

    if feasibility_1:
        if feasibility_2:
            if feasibility_3:
                if violation_1 < violation_2:
                    return members[0] if violation_1 < violation_3 else members[2]
                else:
                    return members[1] if violation_2 < violation_3 else members[2]
            else:
                return members[0] if violation_1 < violation_2 else members[1]
        else:
            if feasibility_3:
                return members[0] if violation_1 < violation_3 else members[2]
            else:
                return members[0]
    else:
        if feasibility_2:
            if feasibility_3:
                return members[1] if violation_2 < violation_3 else members[2]
            else:
                return members[1]
        else:
            if feasibility_3:
                return members[2]
            else:
                if violation_1 < violation_2:
                    return members[0] if violation_1 < violation_3 else members[0]
                else:
                    return members[1] if violation_2 < violation_3 else members[2]


def code_compute_epsilon(curr_gen: int, max_gen: int, e0: float, p: float, cp: float) -> float:
    if curr_gen / max_gen <= p:
        return e0 * math.pow((1 - curr_gen / max_gen), cp)
    else:
        return 0


def code_constraint_method(base_member: Member, trial_member: Member, epsilon: float) -> Member:
    new_member = copy.deepcopy(base_member)

    base_fitness_val = base_member.fitness_value
    trial_fitness_val = trial_member.fitness_value
    for i in range(len(base_member.chromosomes)):
        base_constr_val = code_constraint_val(base_member.chromosomes[i])
        trial_constr_val = code_constraint_val(trial_member.chromosomes[i])

        if base_constr_val < epsilon and trial_constr_val < epsilon:
            if base_fitness_val < trial_fitness_val:
                new_member.chromosomes[i].real_value = base_member.chromosomes[i].real_value
        elif base_constr_val == trial_constr_val:
            if base_fitness_val < trial_fitness_val:
                new_member.chromosomes[i].real_value = base_member.chromosomes[i].real_value
        else:
            if base_constr_val < trial_constr_val:
                new_member.chromosomes[i].real_value = base_member.chromosomes[i].real_value
            else:
                new_member.chromosomes[i].real_value = trial_member.chromosomes[i].real_value

    return new_member
