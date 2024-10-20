from tqdm import tqdm
import time
from DET.DETAlgs.base import BaseAlg
from DET.DETAlgs.data.alg_data import MGDEData
from DET.DETAlgs.methods.methods_de import binomial_crossing, selection
from DET.DETAlgs.methods.methods_mgde import mgde_mutation, mgde_adapt_threshold
from DET.models.enums.boundary_constrain import fix_boundary_constraints
from DET.helpers.metric_helper import MetricHelper
from DET.helpers.database_helper import get_table_name, format_individuals

"""
    MGDE

    Links:
    https://link.springer.com/article/10.1007/s10479-022-04641-3

    References:
    Zouache, D., Ben Abdelaziz, F. MGDE: a many-objective guided differential evolution with strengthened dominance 
    relation and bi-goal evolution. Ann Oper Res (2022). https://doi.org/10.1007/s10479-022-04641-3 
"""


class MGDE(BaseAlg):
    def __init__(self, params: MGDEData, db_conn=None, db_auto_write=False):
        super().__init__(MGDE.__name__, params, db_conn, db_auto_write)

        self.mutation_factor_f = params.mutation_factor_f
        self.mutation_factor_k = params.mutation_factor_k
        self.crossover_rate = params.crossover_rate
        self.threshold = params.threshold
        self.mu = params.mu
        self.generation = 1

    def next_epoch(self):
        # New population after mutation
        v_pop = mgde_mutation(self._pop, self.generation, self.num_of_epochs, self.mutation_factor_f,
                              self.mutation_factor_k)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = binomial_crossing(self._pop, v_pop, cr=self.crossover_rate)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        mgde_adapt_threshold(new_pop, self.threshold, self.mu, self._function.eval)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
