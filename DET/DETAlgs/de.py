from DET.DETAlgs.base import BaseAlg
from DET.DETAlgs.data.alg_data import DEData
from DET.DETAlgs.methods.methods_de import mutation, binomial_crossing, selection
from DET.models.enums.boundary_constrain import fix_boundary_constraints


class DE(BaseAlg):
    def __init__(self, params: DEData, db_conn=None, db_auto_write=False):
        super().__init__(DE.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr

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

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
