import math
import random
import numpy as np
import copy

from src.algorithms.methods.default_de import binomial_crossing_ind
from src.models.member import Member
from src.models.population import Population


def pbx_mutation_ind(base_member: Member, gr_best: Member, member1: Member, member2: Member, f_i):
    """
        Formula:
    """
    new_member = copy.deepcopy(base_member)

    factor = gr_best.chromosomes - base_member.chromosomes + member1.chromosomes - member2.chromosomes
    new_member.chromosomes = base_member.chromosomes + factor * f_i
    return new_member


def pbx_mutation(population: Population, config, f_m, q=0.15):
    num_elements = int(population.size * q)
    best_proc_members = population.get_best_members(population.size)[:num_elements]

    new_members = []
    f_success_set = set()
    for _ in range(population.size):
        gr_best_member = random.sample(best_proc_members.tolist(), 1)[0]

        # Make sure to not select gr_best_member
        id_to_remove = id(gr_best_member)
        members_without_best = [member for member in population.members if id(member) != id_to_remove]

        selected_members = random.sample(members_without_best, 3)

        # Getting f_i from cauchy distribution
        f_i = -1
        while f_i <= 0.0 or f_i > 1.0:
            f_i = np.random.standard_cauchy() * 0.1 + f_m

        new_member = pbx_mutation_ind(selected_members[0], gr_best_member, selected_members[1], selected_members[2], f_i)

        # check if f_i is successful, compare to target vector
        new_member.calculate_fitness_fun(lambda params: config.function.eval(params))
        if new_member > selected_members[0]:
            f_success_set.add(f_i)

        new_members.append(new_member)

    new_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population, f_success_set


def p_best_crossover(origin_population: Population, mutated_population: Population, config, cr_m, curr_gen, max_gen):
    if origin_population.size != mutated_population.size:
        print("Binomial_crossing: populations have different sizes")
        return None

    p = calculate_p_value(origin_population.size, curr_gen, max_gen)
    best_proc_members = origin_population.get_best_members(origin_population.size)[:p]

    new_members = []
    cr_success_set = set()
    for i in range(origin_population.size):
        # Select random vector from best from origin population
        origin_rand_best_vector = random.sample(best_proc_members.tolist(), 1)[0]

        # Getting f_i from Gaussian distribution
        cr_i = -1
        while cr_i < 0.0 or cr_i > 1:
            cr_i = np.random.normal(loc=cr_m, scale=0.1)

        new_member = binomial_crossing_ind(origin_rand_best_vector, mutated_population.members[i], cr_i)

        # check if cr_i is successful, compare to origin_rand_best_vector
        new_member.calculate_fitness_fun(lambda params: config.function.eval(params))
        if new_member > origin_rand_best_vector:
            cr_success_set.add(cr_i)

        new_members.append(new_member)

    new_population = Population(
        interval=origin_population.interval,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population, cr_success_set


def calculate_p_value(pop_size, curr_gen, max_gen):
    half_pop_size = pop_size / 2
    gen_factor = 1 - ((curr_gen - 1) / max_gen)
    return math.ceil(half_pop_size * gen_factor)


def calculate_fm(f_m, f_success_set: set, n=1.5):
    w_f = 0.8 + 0.2 * random.uniform(0, 1)
    mean_pow = 0
    for f in f_success_set:
        mean_pow += (f**n / len(f_success_set)) ** 1 / n

    return w_f * f_m + (1 - w_f) * mean_pow


def calculate_crm(cr_m, cr_success_set: set, n=1.5):
    w_cr = 0.9 + 0.1 * random.uniform(0, 1)
    mean_pow = 0
    for cr in cr_success_set:
        mean_pow += (cr ** n / len(cr_success_set)) ** 1 / n

    return w_cr * cr_m + (1 - w_cr) * mean_pow
