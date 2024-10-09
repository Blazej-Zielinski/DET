import random

from DET.DETAlgs.base import BaseAlg
from DET.DETAlgs.data.alg_data import EIDEData
from DET.DETAlgs.methods.methods_de import mutation, binomial_crossing, selection
from DET.DETAlgs.methods.methods_eide import eide_adopt_parameters
from DET.models.enums.boundary_constrain import fix_boundary_constraints
from DET.models.fitness_function import FitnessFunctionOpfunu
from DET.models.enums.optimization import OptimizationType
from DET.models.enums.boundary_constrain import BoundaryFixing

import opfunu.cec_based.cec2014 as opf


class EIDE(BaseAlg):
    """
        Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6390324&tag=1
    """

    def __init__(self, params: EIDEData = None, db_conn="Differential_evolution.db", db_auto_write=False):
        fitness_fun_opf = FitnessFunctionOpfunu(
            func_type=opf.F82014,
            ndim=10
        )

        if params is None:
            params = EIDEData(
                epoch=100,
                population_size=100,
                dimension=10,
                lb=[-5, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                ub=[5, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                mode=OptimizationType.MINIMIZATION,
                boundary_constraints_fun=BoundaryFixing.RANDOM,
                function=fitness_fun_opf,
                log_population=True,
                crossover_rate_min=0.2,
                crossover_rate_max=0.8
            )

        super().__init__(EIDE.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = random.uniform(0, 0.6)
        self.crossover_rate = params.crossover_rate_min
        self.crossover_rate_min = params.crossover_rate_min
        self.crossover_rate_max = params.crossover_rate_max
        self.generation = None

    def next_epoch(self):
        # New population after mutation
        v_pop = mutation(self._pop, f=self.mutation_factor)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = binomial_crossing(self._pop, v_pop, cr=self.crossover_rate)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        self.mutation_factor, self.crossover_rate = eide_adopt_parameters(self.crossover_rate_min,
                                                                          self.crossover_rate_max, self._epoch_number,
                                                                          self.num_of_epochs)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
