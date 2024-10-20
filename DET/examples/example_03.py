import DET
import opfunu.cec_based.cec2014 as opf

if __name__ == "__main__":
    fitness_fun_opf = DET.FitnessFunctionOpfunu(
        func_type=opf.F82014,
        ndim=10
    )

    params = DET.MGDEData(
        epoch=10,
        population_size=10,
        dimension=10,
        lb=[-5,-100,-100,-100,-100,-100,-100,-100,-100,-100],
        ub=[5,100,100,100,100,100,100,100,100,100],
        mode=DET.OptimizationType.MINIMIZATION,
        boundary_constraints_fun=DET.BoundaryFixing.RANDOM,
        function=fitness_fun_opf,
        crossover_rate=0.8,
        log_population=True,
    )
    params.parallel_processing = ['thread', 5]

    default2 = DET.MGDE(params, db_conn="Differential_evolution.db", db_auto_write=False)
    results = default2.run()
    default2.write_results_to_database(results.epoch_metrics)