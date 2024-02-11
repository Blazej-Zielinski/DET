import numpy as np

from diffEvoLib.diffEvoAlgs.base import BaseDiffEvoAlg
from diffEvoLib.diffEvoAlgs.data.alg_data import AdaptiveParamsData
from diffEvoLib.diffEvoAlgs.methods.methods_adaptive_params import ad_mutation, ad_binomial_crossing, ad_selection
from diffEvoLib.models.enums.boundary_constrain import fix_boundary_constraints


class AdaptiveParams(BaseDiffEvoAlg):
    """
    Source: https://ieeexplore.ieee.org/abstract/document/4730987
    """

    def __init__(self, params: AdaptiveParamsData, db_conn=None, db_auto_write=False):
        super().__init__(AdaptiveParams.__name__, params, db_conn, db_auto_write)

        # class specific
        self._f_arr = np.random.uniform(size=self.population_size)
        self._cr_arr = np.random.uniform(size=self.population_size)
        self.prob_f = params.prob_f
        self.prob_cr = params.prob_cr

    def next_epoch(self):
        f_arr, cr_arr, prob_f, prob_cr = (self._f_arr, self._cr_arr, self.prob_f, self.prob_cr)

        # New population after mutation
        v_pop = ad_mutation(self._pop, f_arr)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = ad_binomial_crossing(self._pop, v_pop, cr_arr)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval)

        # Select new population
        new_pop, f_arr, cr_arr = ad_selection(self._pop, u_pop, f_arr, cr_arr, prob_f, prob_cr)

        # Override data
        self._pop = new_pop
        self._f_arr = f_arr
        self._cr_arr = cr_arr
        self.prob_f = prob_f
        self.prob_cr = prob_cr

        self._epoch_number += 1
