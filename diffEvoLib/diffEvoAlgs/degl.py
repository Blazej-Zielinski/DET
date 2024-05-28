import time
from tqdm import tqdm

from diffEvoLib.diffEvoAlgs.base import BaseDiffEvoAlg
from diffEvoLib.diffEvoAlgs.data.alg_data import DEGLData
from diffEvoLib.diffEvoAlgs.methods.methods_default import binomial_crossing, selection
from diffEvoLib.diffEvoAlgs.methods.methods_degl import degl_mutation, degl_adapt_weight
from diffEvoLib.models.enums.boundary_constrain import fix_boundary_constraints
from diffEvoLib.helpers.metric_helper import MetricHelper


class DEGL(BaseDiffEvoAlg):
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
        u_pop.update_fitness_values(self._function.eval)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1

    def run(self):
        if not self._is_initialized:
            print(f"{self.name} diff evo not initialized.")
            return

        # Calculate metrics
        epoch_metrics = []
        epoch_metric = MetricHelper.calculate_metrics(self._pop, 0.0, epoch=-1)
        epoch_metrics.append(epoch_metric)

        start_time = time.time()
        for epoch in tqdm(range(self.num_of_epochs), desc=f"{self.name}", unit="epoch"):
            self.next_epoch()
            self.weight = degl_adapt_weight(epoch, self.num_of_epochs)

            # Calculate metrics
            epoch_metric = MetricHelper.calculate_metrics(self._pop, start_time, epoch=epoch)
            epoch_metrics.append(epoch_metric)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Function: {self._function.name}, Dimension: {self.nr_of_args},'
              f' Execution time: {execution_time} seconds')

        if self._database is not None and self.db_auto_write:
            self.write_results_to_database(epoch_metrics)
