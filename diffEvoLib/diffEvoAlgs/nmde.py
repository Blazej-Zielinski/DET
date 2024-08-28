import numpy as np

from diffEvoLib.diffEvoAlgs.base import BaseDiffEvoAlg
from diffEvoLib.diffEvoAlgs.data.alg_data import NMDEData
from diffEvoLib.diffEvoAlgs.methods.methods_novel_modified import nm_mutation, nm_selection, nm_calculate_fm_crm, \
    nm_binomial_crossing, nm_update_f_cr
from diffEvoLib.models.enums.boundary_constrain import fix_boundary_constraints


class NMDE(BaseDiffEvoAlg):
    """
    Source: https://www.sciencedirect.com/science/article/pii/S0898122111000460#s000015
    """

    def __init__(self, params: NMDEData, db_conn=None, db_auto_write=False):
        super().__init__(NMDE.__name__, params, db_conn, db_auto_write)

        self.delta_f = params.delta_f
        self.delta_cr = params.delta_cr
        self.sp = params.sp
        self._flags = np.zeros(self.population_size)
        self._f_arr = np.random.uniform(size=self.population_size, low=0, high=2)
        self._cr_arr = np.random.uniform(size=self.population_size, low=0, high=1)
        self._f_set = set()
        self._cr_set = set()

    def next_epoch(self):
        delta_f, delta_cr, sp, flags, f_arr, cr_arr, f_set, cr_set = (
            self.delta_f, self.delta_cr, self.sp, self._flags,
            self._f_arr, self._cr_arr, self._f_set, self._cr_set
        )

        # New population after mutation
        v_pop = nm_mutation(self._pop, f_arr)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = nm_binomial_crossing(self._pop, v_pop, cr_arr)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval)

        # Select new population
        new_pop, better_members_indexes = nm_selection(self._pop, u_pop)

        for i in range(self.population_size):
            if i in better_members_indexes:
                f_set.add(f_arr[i])
                cr_set.add(cr_arr[i])
            else:
                flags[i] += 1

        f_m, cr_m = nm_calculate_fm_crm(f_set, cr_set)

        for i in range(self.population_size):
            if flags[i] == sp:
                if f_set != set() and cr_set != set():
                    f_arr[i], cr_arr[i] = nm_update_f_cr(f_m, cr_m, delta_f, delta_cr)
                else:
                    f_arr[i] = np.random.uniform(low=0, high=2)
                    cr_arr[i] = np.random.uniform(low=0, high=1)
                flags[i] = 0

        # Override data
        self._pop = new_pop
        self.delta_f = delta_f
        self.delta_cr = delta_cr
        self.sp = sp
        self._flags = flags
        self._f_arr = f_arr
        self._cr_arr = cr_arr
        self._f_set = set()
        self._cr_set = set()

        self._epoch_number += 1
