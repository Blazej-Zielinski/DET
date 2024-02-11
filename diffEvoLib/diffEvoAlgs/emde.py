from diffEvoLib.diffEvoAlgs.base import BaseDiffEvoAlg
from diffEvoLib.diffEvoAlgs.data.alg_data import EmDeData
from diffEvoLib.diffEvoAlgs.methods.methods_default import binomial_crossing, selection
from diffEvoLib.diffEvoAlgs.methods.methods_emde import em_mutation
from diffEvoLib.models.enums.boundary_constrain import fix_boundary_constraints


class EmDe(BaseDiffEvoAlg):
    """
    Source: https://link.springer.com/article/10.1007/s13042-015-0479-6#Sec8
    """

    def __init__(self, params: EmDeData, db_conn=None, db_auto_write=False):
        super().__init__(EmDe.__name__, params, db_conn, db_auto_write)

        self.crossover_rate = params.crossover_rate  # Cr

    def next_epoch(self):
        # Calculate not constant cr depend on generation number
        v_pop = em_mutation(self._pop)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = binomial_crossing(self._pop, v_pop, self.crossover_rate)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
