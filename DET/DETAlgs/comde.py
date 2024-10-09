from DET.DETAlgs.base import BaseAlg
from DET.DETAlgs.data.alg_data import COMDEData
from DET.DETAlgs.methods.methods_comde import calculate_cr, comde_mutation
from DET.DETAlgs.methods.methods_de import binomial_crossing, selection
from DET.models.enums.boundary_constrain import fix_boundary_constraints
from DET.models.fitness_function import FitnessFunctionOpfunu
from DET.models.enums.optimization import OptimizationType
from DET.models.enums.boundary_constrain import BoundaryFixing

import opfunu.cec_based.cec2014 as opf

class COMDE(BaseAlg):
    """
    Source: https://www.sciencedirect.com/science/article/pii/S0020025512000278
    """

    def __init__(self, params: COMDEData = None, db_conn="Differential_evolution.db", db_auto_write=False):
        fitness_fun_opf = FitnessFunctionOpfunu(
            func_type=opf.F82014,
            ndim=10
        )

        if params is None:
            params = COMDEData(
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

        super().__init__(COMDE.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr

    def next_epoch(self):
        # Calculate not constant cr depend on generation number
        cr = calculate_cr(self._epoch_number, self.num_of_epochs)

        # New population after mutation
        v_pop = comde_mutation(self._pop)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = binomial_crossing(self._pop, v_pop, cr=cr)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
