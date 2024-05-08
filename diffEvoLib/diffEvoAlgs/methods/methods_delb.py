import random
import numpy as np
import copy

from diffEvoLib.models.population import Population
from diffEvoLib.models.enums.optimization import OptimizationType
from diffEvoLib.models.enums.mutation import mutation_rand_1


def delb_mutation(population: Population):
    new_members = []

    for _ in range(population.size):
        #  Generate random numer from set (−1, -0.4) ∪ (0.4, 1)
        f = random.uniform(-1, -0.4) if random.random() < 0.5 else random.uniform(0.4, 1)
        selected_members = random.sample(population.members.tolist(), 3)

        new_member = mutation_rand_1(selected_members, f)
        new_members.append(new_member)

    new_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)

    return new_population


def delb_selection(origin_population: Population, modified_population: Population, w: float, fitness_func):
    if origin_population.size != modified_population.size:
        print("Selection: populations have different sizes")
        return None

    if origin_population.optimization != modified_population.optimization:
        print("Selection: populations have different optimization types")
        return None

    best_member = origin_population.get_best_members(1)[0]
    optimization = origin_population.optimization

    new_members = []
    for i in range(origin_population.size):
        if optimization == OptimizationType.MINIMIZATION:
            if origin_population.members[i] <= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
            elif random.uniform(0, 1) < w and modified_population.members[i] > best_member:
                r = best_member - (modified_population.members[i] - best_member)
                r.calculate_fitness_fun(fitness_fun=fitness_func)
                if r < modified_population.members[i]:
                    new_members.append(copy.deepcopy(r))
                else:
                    c = best_member + (modified_population.members[i] - best_member) * 0.5
                    c.calculate_fitness_fun(fitness_fun=fitness_func)
                    if c < modified_population.members[i]:
                        new_members.append(copy.deepcopy(c))
                    else:
                        new_members.append(copy.deepcopy(modified_population.members[i]))
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))

        elif optimization == OptimizationType.MAXIMIZATION:
            if origin_population.members[i] >= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
            elif random.uniform(0, 1) < w and modified_population.members[i] < best_member:
                r = best_member - (modified_population.members[i] - best_member)
                r.calculate_fitness_fun(fitness_fun=fitness_func)
                if r > modified_population.members[i]:
                    new_members.append(copy.deepcopy(r))
                else:
                    c = best_member + 0.5 * (modified_population.members[i] - best_member)
                    c.calculate_fitness_fun(fitness_fun=fitness_func)
                    if c > modified_population.members[i]:
                        new_members.append(copy.deepcopy(c))
                    else:
                        new_members.append(copy.deepcopy(modified_population.members[i]))
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))

    new_population = Population(
        interval=origin_population.interval,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)

    return new_population
