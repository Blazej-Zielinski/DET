from diffEvoLib.diffEvoAlgs.base import BaseDiffEvoAlg
from diffEvoLib.diffEvoAlgs.data.alg_data import AADEData
from diffEvoLib.diffEvoAlgs.methods.methods_aade import aade_mutation, aade_crossing, aade_selection, \
    aade_adapat_parameters
from diffEvoLib.models.enums.boundary_constrain import fix_boundary_constraints


class AADE(BaseDiffEvoAlg):
    def __init__(self, params: AADEData, db_conn=None, db_auto_write=False):
        super().__init__(AADE.__name__, params, db_conn, db_auto_write)

        self.mutation_factors = [[params.mutation_factor, False] for _ in range(params.population_size)]
        self.crossover_rates = [[params.crossover_rate, False] for _ in range(params.population_size)]
        self.tolerance = params.tolerance

    def next_epoch(self):
        # New population after mutation
        v_pop = aade_mutation(self._pop, mutation_factors=self.mutation_factors)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = aade_crossing(self._pop, v_pop, crossover_rates=self.crossover_rates)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval)

        # Select new population
        new_pop = aade_selection(self._pop, u_pop, self.mutation_factors, self.crossover_rates)

        aade_adapat_parameters(self._pop, self.mutation_factors, self.crossover_rates)
        # Override data
        self._pop = new_pop

        self._epoch_number += 1
