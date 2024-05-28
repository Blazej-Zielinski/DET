import copy
import time
from abc import ABC, abstractmethod
from tqdm import tqdm

from diffEvoLib.database.database_connector import SQLiteConnector
from diffEvoLib.diffEvoAlgs.data.alg_data import BaseData
from diffEvoLib.helpers.database_helper import get_table_name, format_individuals
from diffEvoLib.helpers.metric_helper import MetricHelper
from diffEvoLib.models.fitness_function import FitnessFunctionBase
from diffEvoLib.models.population import Population


class BaseDiffEvoAlg(ABC):
    def __init__(self, name, params: BaseData, db_conn=None, db_auto_write=False):
        self.name = name
        self._epoch_number = 0
        self._is_initialized = False

        self.optimum = params.optimum
        self.tolerance = params.tolerance

        self._origin_pop = None
        self._pop = None

        self.num_of_epochs = params.num_of_epochs
        self.population_size = params.population_size
        self.nr_of_args = params.nr_of_args
        self.interval = [params.interval_lower_bound, params.interval_higher_bound]
        self.mode = params.mode
        self.boundary_constraints_fun = params.boundary_constraints_fun

        self._function: FitnessFunctionBase = params.function

        self._database = SQLiteConnector(db_conn) if db_conn is not None else None
        self.db_auto_write = db_auto_write

    @abstractmethod
    def next_epoch(self):
        pass

    def initialize(self):
        if self._is_initialized:
            print(f"{self.name} diff evo already initialized.")
            return

        population = Population(
            interval=self.interval,
            arg_num=self.nr_of_args,
            size=self.population_size,
            optimization=self.mode
        )
        population.generate_population()
        population.update_fitness_values(self._function.eval)

        self._origin_pop = population
        self._pop = copy.deepcopy(population)

        self._is_initialized = True

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
            best_member = self._pop.get_best_members(1)[0]
            if abs(self.optimum - best_member.fitness_value) < self.tolerance:
                break
            self.next_epoch()

            # Calculate metrics
            epoch_metric = MetricHelper.calculate_metrics(self._pop, start_time, epoch=epoch)
            epoch_metrics.append(epoch_metric)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Function: {self._function.name}, Dimension: {self.nr_of_args},'
              f' Execution time: {execution_time} seconds')

        if self._database is not None and self.db_auto_write:
            self.write_results_to_database(epoch_metrics)

        return epoch_metrics

    def write_results_to_database(self, results_data):
        print(f'Writing to Database...')

        # Check if database is present
        if self._database is None:
            print(f"There is not database.")
            return

        # Connect to database
        self._database.connect()

        # Creating table
        table_name = get_table_name(
            func_name=self._function.name,
            alg_name=self.name,
            nr_of_args=self.nr_of_args,
            pop_size=self.population_size
        )
        table_name = self._database.create_table(table_name)

        # Inserting data into database
        formatted_best_individuals = format_individuals(results_data)
        self._database.insert_multiple_best_individuals(table_name, formatted_best_individuals)

        self._database.close()
