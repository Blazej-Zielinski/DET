import numpy as np

from DET.DETAlgs.base import BaseAlg
from DET.DETAlgs.data.alg_data import SADEData
from DET.DETAlgs.methods.methods_sade import sade_mutation, sade_binomial_crossing, sade_selection
from DET.models.enums.boundary_constrain import fix_boundary_constraints
from DET.models.fitness_function import FitnessFunctionOpfunu
from DET.models.enums.optimization import OptimizationType
from DET.models.enums.boundary_constrain import BoundaryFixing

import opfunu.cec_based.cec2014 as opf

class SADE(BaseAlg):
    """
    Source: https://ieeexplore.ieee.org/abstract/document/4730987
    """

    def __init__(self, params: SADEData = None, db_conn="Differential_evolution.db", db_auto_write=False):
        fitness_fun_opf = FitnessFunctionOpfunu(
            func_type=opf.F82014,
            ndim=10
        )

        if params is None:
            params = SADEData(
                epoch=100,
                population_size=100,
                dimension=10,
                lb=[-5, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                ub=[5, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                mode=OptimizationType.MINIMIZATION,
                boundary_constraints_fun=BoundaryFixing.RANDOM,
                function=fitness_fun_opf,
                log_population=True,
                prob_cr=1,
                prob_f=1
            )

        super().__init__(SADE.__name__, params, db_conn, db_auto_write)

        # class specific
        self._f_arr = np.random.uniform(size=self.population_size)
        self._cr_arr = np.random.uniform(size=self.population_size)
        self.prob_f = params.prob_f
        self.prob_cr = params.prob_cr

    def next_epoch(self):
        f_arr, cr_arr, prob_f, prob_cr = (self._f_arr, self._cr_arr, self.prob_f, self.prob_cr)

        # New population after mutation
        v_pop = sade_mutation(self._pop, f_arr)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = sade_binomial_crossing(self._pop, v_pop, cr_arr)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop, f_arr, cr_arr = sade_selection(self._pop, u_pop, f_arr, cr_arr, prob_f, prob_cr)

        # Override data
        self._pop = new_pop
        self._f_arr = f_arr
        self._cr_arr = cr_arr
        self.prob_f = prob_f
        self.prob_cr = prob_cr

        self._epoch_number += 1
