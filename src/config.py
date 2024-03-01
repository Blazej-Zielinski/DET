from src.algorithms.boudary_fixing import BoundaryFixing
from src.enums.optimization import OptimizationType
from src.functions.fun_main import FUNCTIONS, FunctionObject
from src.enums.algorithm import AlgorithmType
from src.enums.strategies import StrategiesEnum
from src.models.strategy import Strategy


class Config:
    def __init__(self):
        self.algorithm_type = AlgorithmType.DEFAULT

        self.num_of_epochs = 10
        self.population_size = 100
        self.nr_of_args = 10
        self.interval = [-100, 100]
        self.mutation_factor = 0.5  # F
        self.crossover_rate = 0.8  # cr
        self.mode = OptimizationType.MINIMIZATION
        self.function = FunctionObject(FUNCTIONS.F1, self.nr_of_args)
        self.boundary_constraints_fun = BoundaryFixing.RANDOM

        self.run_all_functions = False
        self.run_all_args = False
        self.nr_of_args_arr = [10, 20, 30]

        self.mutation_strategies = [StrategiesEnum.RAND_1]
        self.learning_period = 25
        if len(self.mutation_strategies) >= 1:
            self.mutation_strategies = [Strategy(stg, 1 / (len(self.mutation_strategies)))
                                        for stg in self.mutation_strategies]

    def sort_mutation_strategies(self):
        self.mutation_strategies = sorted(self.mutation_strategies, key=lambda strategy: strategy.probability)


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
