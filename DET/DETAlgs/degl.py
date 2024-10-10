from DET.DETAlgs.base import BaseAlg
from DET.DETAlgs.data.alg_data import DEGLData
from DET.DETAlgs.methods.methods_de import binomial_crossing, selection
from DET.DETAlgs.methods.methods_degl import degl_mutation, degl_adapt_weight
from DET.models.enums.boundary_constrain import fix_boundary_constraints

"""
    DEGL

    Links:
    https://ieeexplore.ieee.org/document/5089881/

    References:
    S. Das, A. Abraham, U. K. Chakraborty and A. Konar, 
    "Differential Evolution Using a Neighborhood-Based Mutation Operator," 
    in IEEE Transactions on Evolutionary Computation, vol. 13, no. 3, pp. 526-553, June 2009, 
    doi: 10.1109/TEVC.2008.2009457. 
"""


class DEGL(BaseAlg):
    def __init__(self, params: DEGLData, db_conn=None, db_auto_write=False):
        super().__init__(DEGL.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr
        self.radius = params.radius
        self.weight = 0

    def next_epoch(self):
        # New population after mutation
        v_pop = degl_mutation(self._pop, self.radius, self.mutation_factor, self.weight)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = binomial_crossing(self._pop, v_pop, cr=self.crossover_rate)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        self.weight = degl_adapt_weight(self._epoch_number, self.num_of_epochs)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
