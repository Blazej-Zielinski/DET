from DET.DETAlgs.base import BaseAlg
from DET.DETAlgs.data.alg_data import DELBData
from DET.DETAlgs.methods.methods_delb import delb_mutation, delb_selection
from DET.DETAlgs.methods.methods_de import binomial_crossing
from DET.models.enums.boundary_constrain import fix_boundary_constraints
from DET.models.fitness_function import FitnessFunctionOpfunu
from DET.models.enums.optimization import OptimizationType
from DET.models.enums.boundary_constrain import BoundaryFixing

import opfunu.cec_based.cec2014 as opf


class DELB(BaseAlg):
    """
        Source: https://www.sciencedirect.com/science/article/pii/S037722170500281X#aep-section-id9
    """

    def __init__(self, params: DELBData = None, db_conn="Differential_evolution.db", db_auto_write=False):
        fitness_fun_opf = FitnessFunctionOpfunu(
            func_type=opf.F82014,
            ndim=10
        )

        if params is None:
            params = DELBData(
                epoch=100,
                population_size=100,
                dimension=10,
                lb=[-5, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                ub=[5, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                mode=OptimizationType.MINIMIZATION,
                boundary_constraints_fun=BoundaryFixing.RANDOM,
                function=fitness_fun_opf,
                log_population=True,
                crossover_rate=0.5,
                w_factor=2
            )

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
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = delb_selection(self._pop, u_pop, self.w_factor, self._function.eval)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
