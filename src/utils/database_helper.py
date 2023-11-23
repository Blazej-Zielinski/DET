import json

from src.config import Config
from src.models.member import Member
from src.models.result import Result
from src.models.optimum import Optimum
from src.models.measurement import Measurement


def get_table_name(config: Config):
    table_name = f"{config.function.func_type.name}_{config.algorithm_type.name}_" \
                 f"args{config.nr_of_args}_pop{config.population_size}_" \
                 f"results"
                 #f"f{convert_float_to_string(config.mutation_factor)}_" \
                 #f"cr{convert_float_to_string(config.crossover_rate)}" \

    return table_name


def convert_float_to_string(value):
    string_value = format(value, '.1f')
    if string_value.startswith('0.'):
        string_value = string_value.replace('0.', '0')
    return string_value


def format_individuals(individuals):
    formatted_individuals = []
    for data in individuals:
        epoch = data[0]
        best_member: Member = data[1]
        worst_member: Member = data[2]
        mean = data[3]
        std = data[4]
        exec_time = data[5]
        formatted_individuals.append(
            (
                epoch,
                json.dumps([chromosome.real_value for chromosome in best_member.chromosomes]),
                best_member.fitness_value,
                json.dumps([chromosome.real_value for chromosome in worst_member.chromosomes]),
                worst_member.fitness_value,
                mean,
                std,
                exec_time
            )
        )
    return formatted_individuals


def format_results(rows):
    results = map(lambda row: Result(row), rows)
    return list(results)


def format_optimums(rows):
    results = map(lambda row: Optimum(row), rows)
    return list(results)


def format_measurements(rows):
    results = map(lambda row: Measurement(row), rows)
    return list(results)
