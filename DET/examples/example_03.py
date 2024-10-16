import matplotlib.pyplot as plt
import opfunu.cec_based.cec2014 as opf
from DET import COMDE, DE, SADE
from DET.DETAlgs.data.alg_data import COMDEData, DEData, SADEData
from DET.models.fitness_function import FitnessFunctionOpfunu
from DET.models.enums import optimization, boundary_constrain
def extract_best_fitness(epoch_metrics):
    return [epoch.best_individual.fitness_value for epoch in epoch_metrics]

def run_algorithm(algorithm_class, params, db_conn="Differential_evolution.db", db_auto_write=False):
    algorithm = algorithm_class(params, db_conn=db_conn, db_auto_write=db_auto_write)
    results = algorithm.run()
    return [epoch.best_individual.fitness_value for epoch in results.epoch_metrics]


def plot_fitness_convergence(fitness_results, algorithm_names, num_of_epochs):
    epochs = range(1, num_of_epochs + 1)
    for fitness_values, name in zip(fitness_results, algorithm_names):
        fitness_values = fitness_values[:num_of_epochs]
        plt.plot(epochs, fitness_values, label=name)

    plt.xlabel('Epoch')
    plt.ylabel('Best Fitness Value')
    plt.title('Fitness Convergence Algorithms')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    num_of_epochs = 100

    fitness_fun_opf = FitnessFunctionOpfunu(
        func_type=opf.F82014,
        ndim=10
    )

    params_common = {
        'epoch': num_of_epochs,
        'population_size': 100,
        'dimension': 2,
        'lb': [-32.768, -32.768],
        'ub': [32.768, 32.768],
        'mode': optimization.OptimizationType.MINIMIZATION,
        'boundary_constraints_fun': boundary_constrain.BoundaryFixing.RANDOM,
        'function': fitness_fun_opf,
        'log_population': True,
        'parallel_processing': ['thread', 5]
    }

    params_sade = SADEData(**params_common)
    params_comde = COMDEData(**params_common)
    params_de = DEData(**params_common)

    fitness_sade = run_algorithm(SADE, params_sade)
    fitness_comde = run_algorithm(COMDE, params_comde)
    fitness_de = run_algorithm(DE, params_de)

    fitness_results = [fitness_sade, fitness_comde, fitness_de]
    algorithm_names = ['SADE', 'COMDE', 'DE']

    plot_fitness_convergence(fitness_results, algorithm_names, num_of_epochs)