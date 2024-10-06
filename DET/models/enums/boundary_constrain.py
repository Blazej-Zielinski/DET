from enum import Enum

from DET.models.chromosome import Chromosome
from DET.models.member import Member
from DET.models.population import Population


class BoundaryFixing(Enum):
    CLIPPING = 'clipping'
    REFLECTION = 'reflection'
    RANDOM = 'random'


def get_boundary_constraints_fun(fix_type: BoundaryFixing):
    return {
        BoundaryFixing.CLIPPING: lambda member: boundary_clipping(member),
        BoundaryFixing.REFLECTION: lambda member: boundary_reflection(member),
        BoundaryFixing.RANDOM: lambda member: boundary_random(member),
    }.get(fix_type, lambda: None)


def fix_boundary_constraints(population: Population, fix_type: BoundaryFixing):
    boundary_constraints_fun = get_boundary_constraints_fun(fix_type)

    for member in population.members:
        # Enter if member not in interval
        if not member.is_member_in_interval():
            boundary_constraints_fun(member)


# Strategies for fixing members, when they are beyond boundaries


def boundary_clipping(member: Member):
    """
    Modifies the values of `member` in-place.

    :param member: The member to be modified.
    """
    for chromosome in member.chromosomes:
        if chromosome.real_value > chromosome.interval[1]:
            chromosome.real_value = chromosome.interval[1]
        elif chromosome.real_value < chromosome.interval[0]:
            chromosome.real_value = chromosome.interval[0]


def boundary_reflection(member: Member):
    """
    Modifies the values of `member` in-place.

    :param member: The member to be modified.
    """
    for chromosome in member.chromosomes:
        if chromosome.real_value > chromosome.interval[1]:
            chromosome.real_value = 2 * chromosome.interval[1] - chromosome.real_value
        elif chromosome.real_value < chromosome.interval[0]:
            chromosome.real_value = 2 * chromosome.interval[0] - chromosome.real_value


def boundary_random(member: Member):
    """
    Modifies the values of `member` in-place.

    :param member: The member to be modified.
    """
    for chromosome in member.chromosomes:
        if chromosome.real_value < chromosome.interval[0] or chromosome.real_value > chromosome.interval[1]:
            chromosome.real_value = Chromosome(member.interval).real_value
