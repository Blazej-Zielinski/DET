import copy
import time
from tqdm import tqdm

from DET.DETAlgs.base import BaseAlg
from DET.DETAlgs.data.alg_data import OppBasedData
from DET.DETAlgs.methods.methods_opposition_based import opp_based_keep_best_individuals, \
    opp_based_generation_jumping
from DET.DETAlgs.methods.methods_de import mutation, binomial_crossing, selection
from DET.models.enums.boundary_constrain import fix_boundary_constraints
from DET.models.population import Population
from DET.helpers.metric_helper import MetricHelper


class OppBasedDE(BaseAlg):
    """
        Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4358759
    """

    def __init__(self, params: OppBasedData, db_conn=None, db_auto_write=False):
        super().__init__(OppBasedDE.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr
        self.nfc = 0  # number of function calls
        self.max_nfc = params.max_nfc
        self.jumping_rate = params.jumping_rate

    def next_epoch(self):
        # New population after mutation
        v_pop = mutation(self._pop, f=self.mutation_factor)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = binomial_crossing(self._pop, v_pop, cr=self.crossover_rate)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval)
        self.nfc += self.population_size

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Generation jumping
        if opp_based_generation_jumping(new_pop, self.jumping_rate, self._function.eval):
            self.nfc += self.population_size

        # Override data
        self._pop = new_pop

        self._epoch_number += 1

    def initialize(self):
        if self._is_initialized:
            print(f"{self.name} diff evo already initialized.")
            return

        #  Generate initial population
        population = Population(
            interval=self.interval,
            arg_num=self.nr_of_args,
            size=self.population_size,
            optimization=self.mode
        )
        population.generate_population()
        population.update_fitness_values(self._function.eval)

        opp_based_keep_best_individuals(population, self._function.eval, True)

        self._origin_pop = population
        self._pop = copy.deepcopy(population)

        self._is_initialized = True
        self.nfc = 2 * self.population_size

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
            best_member = self._pop.get_best_members(1)[0]
            if abs(self.optimum - best_member.fitness_value) < self.tolerance or self.nfc > self.max_nfc:
                break
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
