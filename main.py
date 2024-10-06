from time import sleep

import DET
import opfunu.cec_based.cec2014 as opf


def example_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    sleep(0.02)
    return (x1 - 1)**2 + (x2 - 2)**2 + (x3 - 3)**2 + (x4 - 4)**2 + (x5 - 5)**2 + \
           (x6 - 6)**2 + (x7 - 7)**2 + (x8 - 8)**2 + (x9 - 9)**2 + (x10 - 10)**2


if __name__ == "__main__":
    num_of_epochs = 100

    fitness_fun = DET.FitnessFunction(
        func=example_function
    )

    fitness_fun_opf = DET.FitnessFunctionOpfunu(
        func_type=opf.F12014,
        ndim=10
    )

    params = DET.DEData(
        num_of_epochs=num_of_epochs,
        population_size=100,
        nr_of_args=10,
        interval_lower_bound=-100,
        interval_higher_bound=100,
        mode=DET.OptimizationType.MINIMIZATION,
        boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
        function=fitness_fun_opf,
        mutation_factor=0.5,
        crossover_rate=0.8
    )
    # params.optimum = 100.0
    # params.tolerance = 0.1

    # default = DET.Default(params, db_conn="Differential_evolution.db", db_auto_write=True)
    # default.initialize()
    # default.run()

    default2 = DET.DE(params, db_conn="Differential_evolution.db", db_auto_write=False)
    default2.initialize()
    results = default2.run()
    # default2.write_results_to_database(results)
















    # params = DET.BestWorstData(
    #     num_of_epochs=num_of_epochs,
    #     population_size=100,
    #     nr_of_args=10,
    #     interval_lower_bound=-100,
    #     interval_higher_bound=100,
    #     mode=DET.OptimizationType.MINIMIZATION,
    #     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
    #     function=fitness_fun,
    #     mutation_factor=0.5,
    #     crossover_rate=0.8
    # )
    #
    # best_worst = DET.BestWorst(params)
    # best_worst.initialize()
    # best_worst.run()
    #
    # params = DET.RandomLocationsData(
    #     num_of_epochs=num_of_epochs,
    #     population_size=100,
    #     nr_of_args=10,
    #     interval_lower_bound=-100,
    #     interval_higher_bound=100,
    #     mode=DET.OptimizationType.MINIMIZATION,
    #     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
    #     function=fitness_fun,
    #     mutation_factor=0.5,
    #     crossover_rate=0.8
    # )
    #
    # random_locations = DET.RandomLocations(params)
    # random_locations.initialize()
    # random_locations.run()
    #
    # params = DET.NovelModifiedData(
    #     num_of_epochs=num_of_epochs,
    #     population_size=100,
    #     nr_of_args=10,
    #     interval_lower_bound=-100,
    #     interval_higher_bound=100,
    #     mode=DET.OptimizationType.MINIMIZATION,
    #     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
    #     function=fitness_fun,
    #     delta_f=0.2,
    #     delta_cr=0.1,
    #     sp=50
    # )
    #
    # novel_modified = DET.NovelModified(params)
    # novel_modified.initialize()
    # novel_modified.run()
    #
    # params = DET.AdaptiveParamsData(
    #     num_of_epochs=num_of_epochs,
    #     population_size=100,
    #     nr_of_args=10,
    #     interval_lower_bound=-100,
    #     interval_higher_bound=100,
    #     mode=DET.OptimizationType.MINIMIZATION,
    #     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
    #     function=fitness_fun,
    #     prob_f=0.1,
    #     prob_cr=0.1
    # )
    #
    # adaptive = DET.AdaptiveParams(params)
    # adaptive.initialize()
    # adaptive.run()
    #
    # params = DET.EmDeData(
    #     num_of_epochs=num_of_epochs,
    #     population_size=100,
    #     nr_of_args=10,
    #     interval_lower_bound=-100,
    #     interval_higher_bound=100,
    #     mode=DET.OptimizationType.MINIMIZATION,
    #     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
    #     function=fitness_fun,
    #     crossover_rate=0.8
    # )
    #
    # emde = DET.EmDe(params)
    # emde.initialize()
    # emde.run()
    #
    # params = DET.ScalingParamsData(
    #     num_of_epochs=num_of_epochs,
    #     population_size=100,
    #     nr_of_args=10,
    #     interval_lower_bound=-100,
    #     interval_higher_bound=100,
    #     mode=DET.OptimizationType.MINIMIZATION,
    #     boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
    #     function=fitness_fun
    # )
    #
    # scaling_params = DET.ScalingParams(params)
    # scaling_params.initialize()
    # scaling_params.run()





    # def evaluate_random_sequence(_):
    #     return function.evaluate([random.uniform(-10, 10) for _ in range(10)])
    #
    # function = FUNCTIONS.F1.value(ndim=10)
    #
    # start_time = time.time()
    # for x in range(1000 * 100):
    #     function.evaluate([random.uniform(-10, 10) for _ in range(10)])
    #
    # end_time = time.time()
    #
    # print(end_time - start_time)
    #
    # start_time = time.time()
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     # Map the workload (range of iterations) to threads
    #     results = executor.map(evaluate_random_sequence, range(1000 * 100))
    #
    # end_time = time.time()
    #
    # print(end_time - start_time)