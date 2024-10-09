from DET.DETAlgs.base import BaseAlg
from DET.DETAlgs.data.alg_data import IDEData
from DET.DETAlgs.methods.methods_de import selection, mutation
from DET.DETAlgs.methods.methods_ide import ide_get_f, ide_get_cr, ide_binomial_crossing
from DET.models.enums.boundary_constrain import fix_boundary_constraints
from DET.models.fitness_function import FitnessFunctionOpfunu
from DET.models.enums.optimization import OptimizationType
from DET.models.enums.boundary_constrain import BoundaryFixing

import opfunu.cec_based.cec2014 as opf

class IDE(BaseAlg):
    """
    Source: https://www.scirp.org/journal/paperinformation.aspx?paperid=96749
    """

    def __init__(self, params: IDEData = None, db_conn="Differential_evolution.db", db_auto_write=False):
        fitness_fun_opf = FitnessFunctionOpfunu(
            func_type=opf.F82014,
            ndim=10
        )

        if params is None:
            params = IDEData(
                epoch=100,
                population_size=100,
                dimension=10,
                lb=[-5, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                ub=[5, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                mode=OptimizationType.MINIMIZATION,
                boundary_constraints_fun=BoundaryFixing.RANDOM,
                function=fitness_fun_opf,
                log_population=True
            )

        super().__init__(IDE.__name__, params, db_conn, db_auto_write)

    def next_epoch(self):
        # Calculate F and CR
        f = ide_get_f(self._epoch_number, self.num_of_epochs)
        cr_arr = ide_get_cr(self._pop)

        # New population after mutation
        v_pop = mutation(self._pop, f)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = ide_binomial_crossing(self._pop, v_pop, cr_arr)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
