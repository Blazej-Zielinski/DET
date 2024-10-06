import numpy as np
from math import floor
import random
import copy

from DET.models.population import Population
from DET.models.member import Member

def issde_mutation(population: Population, eta: float, f: float) -> Population:
    pop_members_list = population.members.tolist()

    n = random.randint(0, population.size - 1)
    m = random.randint(n + 1, population.size - 1)
    l = random.randint(m + 1, population.size - 1)

    x_n = population.members[n]
    x_m = population.members[m]
    x_l = population.members[l]

    a = x_m.chromosomes - x_n.chromosomes
