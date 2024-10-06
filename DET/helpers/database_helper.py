import json
import typing

from DET.helpers.metric_helper import Metric
from DET.models.member import Member


def get_table_name(func_name, alg_name, nr_of_args, pop_size):
    table_name = f"{func_name}_{alg_name}_" \
                 f"args{nr_of_args}_pop{pop_size}_" \
                 f"results"

    return table_name


def format_individuals(individuals: typing.List[Metric]):
    formatted_individuals = []
    for data in individuals:
        epoch = data.epoch
        best_member: Member = data.best_individual
        worst_member: Member = data.worst_individual
        mean = data.population_mean
        std = data.population_std
        exec_time = data.execution_time
        population = [str(member) for member in data.population]

        formatted_individuals.append(
            (
                epoch,
                json.dumps([chromosome.real_value for chromosome in best_member.chromosomes]),
                best_member.fitness_value,
                json.dumps([chromosome.real_value for chromosome in worst_member.chromosomes]),
                worst_member.fitness_value,
                mean,
                std,
                exec_time,
                json.dumps(population)
            )
        )
    return formatted_individuals
