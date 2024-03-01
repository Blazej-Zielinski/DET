import copy
import time

from src.models.population import Population
from src.enums.algorithm import get_algorithm
from src.config import Config


def    diff_evo_alg(pop: Population, config: Config, start_time=None):
    best_individuals = []
    temp_pop = copy.deepcopy(pop)

    # Getting proper algorithm
    algorithm, initialize_alg_vars = get_algorithm(config.algorithm_type)
    algorithm_vars = initialize_alg_vars(config) if initialize_alg_vars is not None else None

    data = calculate_results(temp_pop, start_time, -1)
    best_individuals.append(data)

    for epoch in range(config.num_of_epochs):
        # Applying selected algorithm
        new_pop, algorithm_vars = algorithm(temp_pop, config, epoch, algorithm_vars)

        # Setting new population
        temp_pop = new_pop

        # Calculate results
        data = calculate_results(temp_pop, start_time, epoch)
        best_individuals.append(data)

    return best_individuals


def calculate_results(temp_pop, start_time, epoch):
    sorted_members = temp_pop.get_best_members(temp_pop.size)
    # Get best individual after epoch
    best_inv = sorted_members[0]
    worst_inv = sorted_members[-1]

    # Metrics
    pop_mean = temp_pop.mean()
    pop_std = temp_pop.std()

    end_time = time.time()
    execution_time = end_time - start_time

    data = (epoch + 1, best_inv, worst_inv, pop_mean, pop_std, execution_time)
    print(f"Epoch {epoch + 1}")
    print("Best member:")
    print(best_inv)

    return data
