from DET.DETAlgs.base import BaseAlg
from DET.DETAlgs.data.alg_data import EMDEData
from DET.DETAlgs.methods.methods_de import binomial_crossing, selection
from DET.DETAlgs.methods.methods_emde import em_mutation
from DET.models.enums.boundary_constrain import fix_boundary_constraints
from DET.models.fitness_function import FitnessFunctionOpfunu
from DET.models.enums.optimization import OptimizationType
from DET.models.enums.boundary_constrain import BoundaryFixing

import opfunu.cec_based.cec2014 as opf

class EMDE(BaseAlg):
    """
    Source: https://link.springer.com/article/10.1007/s13042-015-0479-6#Sec8
    """

    def __init__(self, params: EMDEData = None, db_conn="Differential_evolution.db", db_auto_write=False):
        fitness_fun_opf = FitnessFunctionOpfunu(
            func_type=opf.F82014,
            ndim=10
        )

        if params is None:
            params = EMDEData(
                epoch=100,
                population_size=100,
                dimension=10,
                lb=[-5, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                ub=[5, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                mode=OptimizationType.MINIMIZATION,
                boundary_constraints_fun=BoundaryFixing.RANDOM,
                function=fitness_fun_opf,
                log_population=True,
                crossover_rate=0.5
            )

        super().__init__(EMDE.__name__, params, db_conn, db_auto_write)

        self.crossover_rate = params.crossover_rate  # Cr

    def next_epoch(self):
        # Calculate not constant cr depend on generation number
        v_pop = em_mutation(self._pop)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = binomial_crossing(self._pop, v_pop, self.crossover_rate)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
