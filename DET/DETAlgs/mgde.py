from tqdm import tqdm
import time

from DET.DETAlgs.base import BaseAlg
from DET.DETAlgs.data.alg_data import MGDEData
from DET.DETAlgs.methods.methods_de import binomial_crossing, selection
from DET.DETAlgs.methods.methods_mgde import mgde_mutation, mgde_adapt_threshold
from DET.models.enums.boundary_constrain import fix_boundary_constraints
from DET.helpers.metric_helper import MetricHelper


class MGDE(BaseAlg):
    def __init__(self, params: MGDEData, db_conn=None, db_auto_write=False):
        super().__init__(MGDE.__name__, params, db_conn, db_auto_write)

        self.mutation_factor_f = params.mutation_factor_f
        self.mutation_factor_k = params.mutation_factor_k
        self.crossover_rate = params.crossover_rate
        self.threshold = params.threshold
        self.mu = params.mu
        self.generation = None

    def next_epoch(self):
        # New population after mutation
        v_pop = mgde_mutation(self._pop, self.generation, self.num_of_epochs, self.mutation_factor_f, self.mutation_factor_k)

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

    def run(self):
        if not self._is_initialized:
            print(f"{self.name} diff evo not initialized.")
            return

        # Calculate metrics
        epoch_metrics = []
        epoch_metric = MetricHelper.calculate_metrics(self._pop, 0.0, -1, self.log_population)
        epoch_metrics.append(epoch_metric)

        start_time = time.time()
        for epoch in tqdm(range(self.num_of_epochs), desc=f"{self.name}", unit="epoch"):
            self.generation = epoch
            self.next_epoch()

            # Calculate metrics
            epoch_metric = MetricHelper.calculate_metrics(self._pop, start_time, epoch, self.log_population)
            epoch_metrics.append(epoch_metric)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Function: {self._function.name}, Dimension: {self.nr_of_args},'
              f' Execution time: {execution_time} seconds')

        if self._database is not None and self.db_auto_write:
            self.write_results_to_database(epoch_metrics)

        return epoch_metrics