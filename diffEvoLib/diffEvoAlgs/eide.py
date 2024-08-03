import random
from tqdm import tqdm
import time

from diffEvoLib.diffEvoAlgs.base import BaseDiffEvoAlg
from diffEvoLib.diffEvoAlgs.data.alg_data import EIDEData
from diffEvoLib.diffEvoAlgs.methods.methods_default import mutation, binomial_crossing, selection
from diffEvoLib.diffEvoAlgs.methods.methods_eide import eide_adopt_parameters
from diffEvoLib.models.enums.boundary_constrain import fix_boundary_constraints
from diffEvoLib.helpers.metric_helper import MetricHelper


class EIDE(BaseDiffEvoAlg):
    """
        Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6390324&tag=1
    """

    def __init__(self, params: EIDEData, db_conn=None, db_auto_write=False):
        super().__init__(EIDE.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = random.uniform(0, 0.6)
        self.crossover_rate = params.crossover_rate_min
        self.crossover_rate_min = params.crossover_rate_min
        self.crossover_rate_max = params.crossover_rate_max
        self.generation = None

    def next_epoch(self):
        # New population after mutation
        v_pop = mutation(self._pop, f=self.mutation_factor)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = binomial_crossing(self._pop, v_pop, cr=self.crossover_rate)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        self.mutation_factor, self.crossover_rate = eide_adopt_parameters(self.crossover_rate_min,
                                                                          self.crossover_rate_max, self._epoch_number,
                                                                          self.num_of_epochs)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
