from diffEvoLib.diffEvoAlgs.base import BaseDiffEvoAlg
from diffEvoLib.diffEvoAlgs.data.alg_data import AADEData
from diffEvoLib.diffEvoAlgs.methods.methods_default import mutation, binomial_crossing
from diffEvoLib.diffEvoAlgs.methods.methods_aade import aade_selection, aade_adapat_parameters
from diffEvoLib.models.enums.boundary_constrain import fix_boundary_constraints


class AADE(BaseDiffEvoAlg):
    def __init__(self, params: AADEData, db_conn=None, db_auto_write=False):
        super().__init__(AADE.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = params.mutation_factor  # F
        self.mutation_factors = []
        self.crossover_rate = params.crossover_rate  # Cr
        self.crossover_rates = []

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
        new_pop = aade_selection(self._pop, u_pop, self.mutation_factors, self.crossover_rates, self.mutation_factor,
                                 self.crossover_rate)

        self.mutation_factor, self.crossover_rate = aade_adapat_parameters(self.mutation_factors, self.crossover_rates)
        # Override data
        self._pop = new_pop

        self._epoch_number += 1
