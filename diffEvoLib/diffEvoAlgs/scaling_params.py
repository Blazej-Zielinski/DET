from diffEvoLib.diffEvoAlgs.base import BaseDiffEvoAlg
from diffEvoLib.diffEvoAlgs.data.alg_data import ScalingParamsData
from diffEvoLib.diffEvoAlgs.methods.methods_default import selection, mutation
from diffEvoLib.diffEvoAlgs.methods.methods_scaling_params import sp_get_f, sp_get_cr, sp_binomial_crossing
from diffEvoLib.models.enums.boundary_constrain import fix_boundary_constraints


class ScalingParams(BaseDiffEvoAlg):
    """
    Source: https://www.scirp.org/journal/paperinformation.aspx?paperid=96749
    """

    def __init__(self, params: ScalingParamsData, db_conn=None, db_auto_write=False):
        super().__init__(ScalingParams.__name__, params, db_conn, db_auto_write)

    def next_epoch(self):
        # Calculate F and CR
        f = sp_get_f(self._epoch_number, self.num_of_epochs)
        cr_arr = sp_get_cr(self._pop)

        # New population after mutation
        v_pop = mutation(self._pop, f)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = sp_binomial_crossing(self._pop, v_pop, cr_arr)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
