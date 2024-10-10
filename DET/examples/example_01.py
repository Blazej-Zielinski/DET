import DET
from DET import COMDE, DERL, EMDE, IDE, NMDE, SADE
from DET.DETAlgs.aade import AADE
from DET.DETAlgs.data.alg_data import AADEData, COMDEData, DEGLData, DELBData, DERLData, EIDEData, EMDEData, IDEData, \
    JADEData, MGDEData, NMDEData, OppBasedData, SADEData
from DET.DETAlgs.degl import DEGL
from DET.DETAlgs.delb import DELB
from DET.DETAlgs.eide import EIDE
from DET.DETAlgs.jade import JADE
from DET.DETAlgs.mgde import MGDE
from DET.DETAlgs.opposition_based import OppBasedDE
from DET.functions import FunctionLoader
from DET.models.fitness_function import BenchmarkFitnessFunction

function_loader = FunctionLoader()
ackley_function = function_loader.get_function(function_name="ackley", n_dimensions=2)

fitness_fun = BenchmarkFitnessFunction(ackley_function)

# params = DET.DEData(
#     epoch=100,
#     population_size=100,
#     dimension=2,
#     lb=[-32.768, -32.768],
#     ub=[32.768, 32.768],
#     mode=DET.OptimizationType.MINIMIZATION,
#     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
#     function=fitness_fun,
#     mutation_factor=0.5,
#     crossover_rate=0.8,
#     log_population=True,
#     parallel_processing = ['thread', 5]
# )
#
# default2 = DET.DE(params, db_conn="Differential_evolution.db", db_auto_write=False)
# results = default2.run()


# params = AADEData(
#     epoch=100,
#     population_size=100,
#     dimension=2,
#     lb=[-32.768, -32.768],
#     ub=[32.768, 32.768],
#     mode=DET.OptimizationType.MINIMIZATION,
#     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
#     function=fitness_fun,
#     mutation_factor=0.5,
#     crossover_rate=0.8,
#     log_population=True,
#     parallel_processing = ['thread', 5]
# )
# default2 = AADE(params, db_conn="Differential_evolution.db", db_auto_write=False)
# results = default2.run()

# params = COMDEData(
#     epoch=100,
#     population_size=100,
#     dimension=2,
#     lb=[-32.768, -32.768],
#     ub=[32.768, 32.768],
#     mode=DET.OptimizationType.MINIMIZATION,
#     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
#     function=fitness_fun,
#     mutation_factor=0.5,
#     crossover_rate=0.8,
#     log_population=True,
#     parallel_processing = ['thread', 5]
# )
# default2 = COMDE(params, db_conn="Differential_evolution.db", db_auto_write=False)
# results = default2.run()


# params = DEGLData(
#     epoch=100,
#     population_size=100,
#     dimension=2,
#     lb=[-32.768, -32.768],
#     ub=[32.768, 32.768],
#     mode=DET.OptimizationType.MINIMIZATION,
#     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
#     function=fitness_fun,
#     mutation_factor=0.5,
#     crossover_rate=0.8,
#     log_population=True,
#     parallel_processing = ['thread', 5]
# )
# default2 = DEGL(params, db_conn="Differential_evolution.db", db_auto_write=False)
# results = default2.run()

# params = DELBData(
#     epoch=100,
#     population_size=100,
#     dimension=2,
#     lb=[-32.768, -32.768],
#     ub=[32.768, 32.768],
#     mode=DET.OptimizationType.MINIMIZATION,
#     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
#     function=fitness_fun,
#     crossover_rate=0.8,
#     log_population=True,
#     parallel_processing = ['thread', 5]
# )
# default2 = DELB(params, db_conn="Differential_evolution.db", db_auto_write=False)
# results = default2.run()

# params = DERLData(
#     epoch=100,
#     population_size=100,
#     dimension=2,
#     lb=[-32.768, -32.768],
#     ub=[32.768, 32.768],
#     mode=DET.OptimizationType.MINIMIZATION,
#     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
#     function=fitness_fun,
#     crossover_rate=0.8,
#     log_population=True,
#     parallel_processing = ['thread', 5]
# )
# default2 = DERL(params, db_conn="Differential_evolution.db", db_auto_write=False)
# results = default2.run()

# params = EIDEData(
#     epoch=100,
#     population_size=100,
#     dimension=2,
#     lb=[-32.768, -32.768],
#     ub=[32.768, 32.768],
#     mode=DET.OptimizationType.MINIMIZATION,
#     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
#     function=fitness_fun,
#     log_population=True,
#     parallel_processing = ['thread', 5]
# )
# default2 = EIDE(params, db_conn="Differential_evolution.db", db_auto_write=False)
# results = default2.run()

# params = EMDEData(
#     epoch=100,
#     population_size=100,
#     dimension=2,
#     lb=[-32.768, -32.768],
#     ub=[32.768, 32.768],
#     mode=DET.OptimizationType.MINIMIZATION,
#     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
#     function=fitness_fun,
#     log_population=True,
#     parallel_processing = ['thread', 5]
# )
#
# default2 = EMDE(params, db_conn="Differential_evolution.db", db_auto_write=False)
# results = default2.run()

# params = IDEData(
#     epoch=100,
#     population_size=100,
#     dimension=2,
#     lb=[-32.768, -32.768],
#     ub=[32.768, 32.768],
#     mode=DET.OptimizationType.MINIMIZATION,
#     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
#     function=fitness_fun,
#     log_population=True,
#     parallel_processing = ['thread', 5]
# )
#
# default2 = IDE(params, db_conn="Differential_evolution.db", db_auto_write=False)
# results = default2.run()

# params = JADEData(
#     epoch=100,
#     population_size=100,
#     dimension=2,
#     lb=[-32.768, -32.768],
#     ub=[32.768, 32.768],
#     mode=DET.OptimizationType.MINIMIZATION,
#     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
#     function=fitness_fun,
#     log_population=True,
#     parallel_processing = ['thread', 5]
# )
#
# default2 = JADE(params, db_conn="Differential_evolution.db", db_auto_write=False)
# results = default2.run()

#To do !
# params = MGDEData(
#     epoch=100,
#     population_size=100,
#     dimension=2,
#     lb=[-32.768, -32.768],
#     ub=[32.768, 32.768],
#     mode=DET.OptimizationType.MINIMIZATION,
#     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
#     function=fitness_fun,
#     log_population=True,
#     parallel_processing = ['thread', 5]
# )
#
# default2 = MGDE(params, db_conn="Differential_evolution.db", db_auto_write=False)
# results = default2.run()

# params = NMDEData(
#     epoch=100,
#     population_size=100,
#     dimension=2,
#     lb=[-32.768, -32.768],
#     ub=[32.768, 32.768],
#     mode=DET.OptimizationType.MINIMIZATION,
#     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
#     function=fitness_fun,
#     log_population=True,
#     parallel_processing = ['thread', 5]
# )
#
# default2 = NMDE(params, db_conn="Differential_evolution.db", db_auto_write=False)
# results = default2.run()


# params = NMDEData(
#     epoch=100,
#     population_size=100,
#     dimension=2,
#     lb=[-32.768, -32.768],
#     ub=[32.768, 32.768],
#     mode=DET.OptimizationType.MINIMIZATION,
#     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
#     function=fitness_fun,
#     log_population=True,
#     parallel_processing = ['thread', 5]
# )
#
# default2 = NMDE(params, db_conn="Differential_evolution.db", db_auto_write=False)
# results = default2.run()

#To do !
# params = OppBasedData(
#     epoch=100,
#     population_size=100,
#     dimension=2,
#     lb=[-32.768, -32.768],
#     ub=[32.768, 32.768],
#     mode=DET.OptimizationType.MINIMIZATION,
#     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
#     function=fitness_fun,
#     log_population=True,
#     parallel_processing = ['thread', 5]
# )
#
# default2 = OppBasedDE(params, db_conn="Differential_evolution.db", db_auto_write=False)
# results = default2.run()


params = SADEData(
    epoch=100,
    population_size=100,
    dimension=2,
    lb=[-32.768, -32.768],
    ub=[32.768, 32.768],
    mode=DET.OptimizationType.MINIMIZATION,
    boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
    function=fitness_fun,
    log_population=True,
    parallel_processing = ['thread', 5]
)

default2 = SADE(params, db_conn="Differential_evolution.db", db_auto_write=False)
results = default2.run()
default2.write_results_to_database(results)