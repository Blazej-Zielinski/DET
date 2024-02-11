from diffEvoLib.diffEvoAlgs.base import BaseDiffEvoAlg
from diffEvoLib.diffEvoAlgs.data.alg_data import RandomLocationsData
from diffEvoLib.diffEvoAlgs.methods.methods_default import binomial_crossing, selection
from diffEvoLib.diffEvoAlgs.methods.methods_random_locations import rl_mutation
from diffEvoLib.models.enums.boundary_constrain import fix_boundary_constraints


class RandomLocations(BaseDiffEvoAlg):
    """
    Source: https://www.sciencedirect.com/science/article/pii/S037722170500281X#aep-section-id9
    """

    def __init__(self, params: RandomLocationsData, db_conn=None, db_auto_write=False):
        super().__init__(RandomLocations.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr

    def next_epoch(self):
        # New population after mutation
        v_pop = rl_mutation(self._pop)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = binomial_crossing(self._pop, v_pop, cr=self.crossover_rate)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
