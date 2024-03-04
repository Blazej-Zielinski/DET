import copy

from enum import Enum
from src.models.member import Member


class StrategiesEnum(Enum):
    RAND_1 = "rand_1"
    RAND_2 = "rand_2"
    BEST_1 = "best_1"
    BEST_2 = "best_2"
    CURRENT_TO_BEST_1 = "current_to_best_1"


def mutation_rand_1(members: list[Member], f: float) -> Member:
    """
        Formula: v_ij = x_r1 + F(x_r2 - x_r3)
    """
    new_member = copy.deepcopy(members[0])
    new_member.chromosomes = members[0].chromosomes + (members[1].chromosomes - members[2].chromosomes) * f
    return new_member


def mutation_best_1(best_member: Member, members: list[Member], f: float) -> Member:
    """
        Formula: v_ij = x_best + F(x_r1 - x_r2)
    """
    new_member = copy.deepcopy(best_member)
    new_member.chromosomes = best_member.chromosomes + (members[0].chromosomes - members[1].chromosomes) * f
    return new_member


def mutation_curr_to_best_1(base_member: Member, best_member: Member, members: list[Member], f: float) -> Member:
    """
        Formula: v_ij = x_base + F(x_best - x_base) + F(x_r1 - x_r2)
    """
    new_member = copy.deepcopy(base_member)
    new_member.chromosomes = (base_member.chromosomes + (best_member.chromosomes - base_member.chromosomes) * f +
                              (members[0].chromosomes - members[1].chromosomes) * f)
    return new_member


def mutation_best_2(best_member: Member, members: list[Member], f: float) -> Member:
    """
        Formula: v_ij = x_best + F(x_r1 - x_r2) + F(x_r3 - x_r4)
    """
    new_member = copy.deepcopy(best_member)
    new_member.chromosomes = (best_member.chromosomes + (members[0].chromosomes - members[1].chromosomes) * f +
                              (members[2].chromosomes - members[3].chromosomes) * f)
    return new_member


def mutation_rand_2(members: list[Member], f: float) -> Member:
    """
        Formula: v_ij = x_best + F(x_r1 - x_r2) + F(x_r3 - x_r4)
    """
    new_member = copy.deepcopy(members[0])
    new_member.chromosomes = (members[0].chromosomes + (members[1].chromosomes - members[2].chromosomes) * f +
                              (members[3].chromosomes - members[4].chromosomes) * f)
    return new_member
