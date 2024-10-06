from DET.DETAlgs.base import BaseDiffEvoAlg
from DET.DETAlgs.data.alg_data import DELBData
from DET.DETAlgs.methods.methods_delb import delb_mutation, delb_selection
from DET.DETAlgs.methods.methods_de import binomial_crossing
from DET.models.enums.boundary_constrain import fix_boundary_constraints


class DELB(BaseDiffEvoAlg):
    """
        Source: https://www.sciencedirect.com/science/article/pii/S037722170500281X#aep-section-id9
    """

    def __init__(self, params: DELBData, db_conn=None, db_auto_write=False):
        super().__init__(DELB.__name__, params, db_conn, db_auto_write)

        self.crossover_rate = params.crossover_rate  # Cr
        self.w_factor = params.w_factor  # w

    def next_epoch(self):
        # New population after mutation
        v_pop = delb_mutation(self._pop)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = binomial_crossing(self._pop, v_pop, cr=self.crossover_rate)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval)

        # Select new population
        new_pop = delb_selection(self._pop, u_pop, self.w_factor, self._function.eval)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
