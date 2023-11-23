import random
import numpy as np
import copy

from src.models.member import Member
from src.models.population import Population


def bi_mutation_ind(base_member: Member, member1: Member, member2: Member, f):
    """
        Formula: v_ij = x_r1 + F(x_r2 - x_r3)
        Formula: v_ij = x_r1 - F(x_r2 - x_r3)
    """
    new_member_1 = copy.deepcopy(base_member)
    new_member_2 = copy.deepcopy(base_member)

    new_member_1.chromosomes = base_member.chromosomes + (member1.chromosomes - member2.chromosomes) * f
    new_member_2.chromosomes = base_member.chromosomes - (member1.chromosomes - member2.chromosomes) * f
    return new_member_1, new_member_2


def bi_mutation(population: Population, f):
    new_members = []
    bidirectional_new_members = []
    for _ in range(population.size):
        selected_members = random.sample(population.members.tolist(), 3)
        new_member, bi_new_member = bi_mutation_ind(selected_members[0], selected_members[1], selected_members[2], f)

        new_members.append(new_member)
        bidirectional_new_members.append(bi_new_member)

    new_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )

    bi_new_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )

    new_population.members = np.array(new_members)
    bi_new_population.members = np.array(bidirectional_new_members)
    return new_population, bi_new_population


def bi_binomial_crossing_ind(org_member: Member, mut_member: Member, bi_mut_member: Member, cr):
    new_member = copy.deepcopy(org_member)
    bi_new_member = copy.deepcopy(bi_mut_member)

    random_numbers = np.random.rand(new_member.args_num)
    mask = random_numbers <= cr

    i_rand = np.random.randint(low=0, high=new_member.args_num)

    for i in range(new_member.args_num):
        if mask[i] or i_rand:
            new_member.chromosomes[i].real_value = mut_member.chromosomes[i].real_value
            bi_new_member.chromosomes[i].real_value = bi_mut_member.chromosomes[i].real_value
        else:
            new_member.chromosomes[i].real_value = org_member.chromosomes[i].real_value
            bi_new_member.chromosomes[i].real_value = org_member.chromosomes[i].real_value

    return new_member, bi_new_member


def bi_binomial_crossing(origin_population: Population, mutated_population: Population, bi_mutated_population: Population, cr):
    if origin_population.size != mutated_population.size != bi_mutated_population.size:
        print("Binomial_crossing: populations have different sizes")
        return None

    new_members = []
    bi_new_members = []
    for i in range(origin_population.size):
        new_member, bi_new_member = bi_binomial_crossing_ind(origin_population.members[i],
                                                             mutated_population.members[i],
                                                             bi_mutated_population.members[i], cr)
        new_members.append(new_member)
        bi_new_members.append(bi_new_member)

    new_population = Population(
        interval=origin_population.interval,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    bi_new_population = Population(
        interval=origin_population.interval,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)
    bi_new_population.members = np.array(bi_new_members)
    return new_population, bi_new_population
