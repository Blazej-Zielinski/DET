from DET.DETAlgs.base import BaseAlg
from DET.DETAlgs.data.alg_data import AADEData
from DET.DETAlgs.methods.methods_aade import aade_mutation, aade_crossing, aade_selection, \
    aade_adapat_parameters
from DET.models.enums.boundary_constrain import fix_boundary_constraints
from DET.models.fitness_function import FitnessFunctionOpfunu
from DET.models.enums.optimization import OptimizationType
from DET.models.enums.boundary_constrain import BoundaryFixing

import opfunu.cec_based.cec2014 as opf

class AADE(BaseAlg):
    """
        Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8819749&tag=1
    """

    def __init__(self, params: AADEData = None, db_conn="Differential_evolution.db", db_auto_write=False):
        fitness_fun_opf = FitnessFunctionOpfunu(
            func_type=opf.F82014,
            ndim=10
        )

        if params is None:
            params = AADEData(
                epoch=100,
                population_size=100,
                dimension=10,
                lb=[-5, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                ub=[5, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                mode=OptimizationType.MINIMIZATION,
                boundary_constraints_fun=BoundaryFixing.RANDOM,
                function=fitness_fun_opf,
                log_population=True,
                mutation_factor=0.5,
                crossover_rate=0.5
            )

        super().__init__(AADE.__name__, params, db_conn, db_auto_write)

        self.mutation_factors = [[params.mutation_factor, False] for _ in range(params.population_size)]
        self.crossover_rates = [[params.crossover_rate, False] for _ in range(params.population_size)]

    def next_epoch(self):
        # New population after mutation
        v_pop = aade_mutation(self._pop, mutation_factors=self.mutation_factors)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = aade_crossing(self._pop, v_pop, crossover_rates=self.crossover_rates)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = aade_selection(self._pop, u_pop, self.mutation_factors, self.crossover_rates)

        aade_adapat_parameters(self._pop, self.mutation_factors, self.crossover_rates)
        # Override data
        self._pop = new_pop

        self._epoch_number += 1
