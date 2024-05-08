import numpy as np
import copy

from diffEvoLib.diffEvoAlgs.base import BaseDiffEvoAlg
from diffEvoLib.diffEvoAlgs.data.alg_data import OppBasedData
from diffEvoLib.diffEvoAlgs.methods.methods_delb import delb_mutation, delb_selection
from diffEvoLib.diffEvoAlgs.methods.methods_default import binomial_crossing
from diffEvoLib.models.enums.boundary_constrain import fix_boundary_constraints
from diffEvoLib.models.population import Population, Member


class OppBasedDE(BaseDiffEvoAlg):
    def __init__(self, params: OppBasedData, db_conn=None, db_auto_write=False):
        super().__init__(OppBasedDE.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr
        self.nfc = params.nfc  # number of function calls
        self.max_nfc = params.max_nfc
        self.jumping_rate = params.jumping_rate

    def initialize(self):
        if self._is_initialized:
            print(f"{self.name} diff evo already initialized.")
            return

        #  Generate initial population
        population = Population(
            interval=self.interval,
            arg_num=self.nr_of_args,
            size=self.population_size,
            optimization=self.mode
        )
        population.generate_population()
        population.update_fitness_values(self._function.eval)

        #  Generate opposite population
        central_point = sum(self.interval)
        opposite_members = []
        for member in population.members:
            new_member = Member(self.name, self.nr_of_args)
            for i in range(self.nr_of_args):
                new_member.chromosomes[i] = (member.chromosomes[i] - central_point) * -1
            opposite_members.append(new_member)

        opposite_population = Population(
            interval=self.interval,
            arg_num=self.nr_of_args,
            size=self.population_size,
            optimization=self.mode
        )
        opposite_population.members = np.array(opposite_members)
        opposite_population.update_fitness_values(self._function.eval)

        pops = np.concatenate((population.members, opposite_population.members))
        sorted_pops_indices = np.argsort([member.fitness_value for member in pops])
        sorted_pops = pops[sorted_pops_indices]
        population.members = sorted_pops[:self.population_size]

        self._origin_pop = population
        self._pop = copy.deepcopy(population)

        self._is_initialized = True
