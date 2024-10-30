import detpy
from detpy import SADE
from detpy.DETAlgs.data.alg_data import SADEData
from detpy.functions import FunctionLoader
from detpy.models.fitness_function import BenchmarkFitnessFunction

function_loader = FunctionLoader()
ackley_function = function_loader.get_function(function_name="ackley", n_dimensions=2)
fitness_fun = BenchmarkFitnessFunction(ackley_function)

params = SADEData(
    epoch=100,
    population_size=100,
    dimension=2,
    lb=[-32.768, -32.768],
    ub=[32.768, 32.768],
    mode=detpy.OptimizationType.MINIMIZATION,
    boundary_constraints_fun=detpy.BoundaryFixing.RANDOM,
    function=fitness_fun,
    log_population=True,
    parallel_processing=['thread', 4]
)

default2 = SADE(params, db_conn="Differential_evolution.db", db_auto_write=False)
results = default2.run()

