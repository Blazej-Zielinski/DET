import detpy
import opfunu.cec_based.cec2014 as opf


def example_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return (x1 - 1)**2 + (x2 - 2)**2 + (x3 - 3)**2 + (x4 - 4)**2 + (x5 - 5)**2 + \
           (x6 - 6)**2 + (x7 - 7)**2 + (x8 - 8)**2 + (x9 - 9)**2 + (x10 - 10)**2


if __name__ == "__main__":
    num_of_epochs = 100

    fitness_fun = detpy.FitnessFunction(
        func=example_function
    )

    fitness_fun_opf = detpy.FitnessFunctionOpfunu(
        func_type=opf.F82014,
        ndim=10
    )

    func = opf.F82014(ndim=10)
    print(func.f_global)
    print(func.x_global)

    params = detpy.DEData(
        epoch=10,
        population_size=10,
        dimension=10,
        lb=[-5,-100,-100,-100,-100,-100,-100,-100,-100,-100],
        ub=[5,100,100,100,100,100,100,100,100,100],
        mode=detpy.OptimizationType.MINIMIZATION,
        boundary_constraints_fun=detpy.BoundaryFixing.RANDOM,
        function=fitness_fun_opf,
        mutation_factor=0.5,
        crossover_rate=0.8,
        log_population=True
    )
    params.parallel_processing = ['thread', 5]
    default2 = detpy.DE(params, db_conn="Differential_evolution.db", db_auto_write=False)
    results = default2.run()







