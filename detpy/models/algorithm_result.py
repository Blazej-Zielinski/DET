from matplotlib import pyplot as plt


class AlgorithmResult:
    def __init__(self, epoch_metrics, avg_fitness, std_fitness, best_solution):
        self.epoch_metrics = epoch_metrics
        self.avg_fitness = avg_fitness
        self.std_fitness = std_fitness
        self.best_solution = best_solution

    def __repr__(self):
        return (f"AlgorithmResult(avg_fitness={self.avg_fitness}, std_fitness={self.std_fitness}, "
                f"best_solution={self.best_solution})")

    def plot_results(self, best_fitness_values, avg_fitness_values, std_fitness_values, num_of_epochs, method_name="Method"):
        epochs = range(1, num_of_epochs + 1)

        plt.figure()
        plt.plot(epochs, best_fitness_values, label="Best Fitness")
        plt.grid(True)

        plt.xlabel('Epoch')
        plt.ylabel('Best Fitness Value')
        plt.title(f'Best Fitness per Epoch - {method_name}')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(epochs, avg_fitness_values, label="Average Fitness", color="orange")
        plt.grid(True)

        plt.xlabel('Epoch')
        plt.ylabel('Average Fitness Value')
        plt.title(f'Average Fitness per Epoch - {method_name}')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(epochs, std_fitness_values, label="Standard Deviation of Fitness", color="green")
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Standard Deviation')
        plt.title(f'Standard Deviation of Fitness per Epoch - {method_name}')
        plt.legend()
        plt.show()
