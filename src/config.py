from src.algorithms.boudary_fixing import BoundaryFixing
from src.enums.optimization import OptimizationType
from src.functions.fun_main import FUNCTIONS, FunctionObject
from src.enums.algorithm import AlgorithmType
from src.enums.strategies import StrategiesEnum


class Config:
    def __init__(self):
        self.algorithm_type = AlgorithmType.JADE

        self.num_of_epochs = 101
        self.population_size = 100
        self.nr_of_args = 10
        self.interval = [-100, 100]
        self.mutation_factor = 0.5  # F should be between [0;2] larger F -> higher diversity
        self.crossover_rate = 0.8  # cr
        self.mode = OptimizationType.MINIMIZATION
        self.function = FunctionObject(FUNCTIONS.F1, self.nr_of_args)
        self.boundary_constraints_fun = BoundaryFixing.RANDOM

        self.best_fitness_value = None
        self.value_to_reach = 10e-8

        self.run_all_functions = False
        self.run_all_args = False
        self.nr_of_args_arr = [10, 20, 30]

        if self.algorithm_type == AlgorithmType.SELF_ADAPTIVE:
            self.mutation_factor_mean = 0.5
            self.mutation_factor_std = 0.3
            self.mutation_factor_low = 0
            self.mutation_factor_high = 2
            self.mutation_learning_period = 50
            self.mutation_strategies = [StrategiesEnum.RAND_1, StrategiesEnum.CURRENT_TO_BEST_1]

            self.crossover_rate_mean = 0.5
            self.crossover_rate_std = 0.1
            self.crossover_reinit_period = 5
            self.crossover_learning_period = 25

        if self.algorithm_type == AlgorithmType.JADE:
            self.mutation_factor_mean = 0.5
            self.mutation_factor_std = 0.1
            self.mutation_factor_low = 0
            self.mutation_factor_high = 1

            self.crossover_rate_mean = 0.5
            self.crossover_rate_std = 0.1
            self.crossover_rate_low = 0
            self.crossover_rate_high = 1

            self.jade_c = 0.1  # (constant between (0,1)) describes the rate of parameter adaptation
            self.jade_p = 0.1  # (constant between (0,1)) describes the greediness of the mutation

        if self.algorithm_type == AlgorithmType.OPPOSITION_BASED:
            self.nfc = 0
            self.max_nfc = 10e6
            self.jumping_rate = 0.3
            self.mutation_strategy = StrategiesEnum.RAND_1

    def set_mutation_factor_mean(self, new_mean: float):
        self.mutation_factor_mean = new_mean

    def set_crossover_rate_mean(self, new_mean: float):
        self.crossover_rate_mean = new_mean


class DatabaseConfig:
    def __init__(self):
        self.database = "Differential_evolution.db"


class MultipleDatabaseConfig:
    def __init__(self):
        self.prefix = "xyz_databases/Differential_evolution_"
        self.databases_postfixes = ["Default", "BestWorst", "RandomLocations", "EMDE", "AdaptiveParams",
                                    "NovelModified", "ScalingParams"]
        self.databases = [f"{self.prefix}{postfix}.db" for postfix in self.databases_postfixes]

        self.optima = [i * 100.0 for i in range(1, 31)]
