import time
from numpy.random import uniform
from opfunu.utils import operator

from src.algorithms.main_alg import diff_evo_alg
from src.config import Config, DatabaseConfig
from src.database.connector import SQLiteConnector
from src.functions.fun_main import FunctionObject, FUNCTIONS
from src.functions.utlis import plot_fun, plot_fun_formula
from src.models.population import Population
from src.utils.database_helper import get_table_name, format_individuals


def run_de(config, fun=None):
    config.function = config.function if fun is None else fun

    S_pop = Population(
        interval=config.interval,
        arg_num=config.nr_of_args,
        size=config.population_size,
        optimization=config.mode
    )
    S_pop.generate_population()
    S_pop.update_fitness_values(lambda params: config.function.eval(params))

    # Start algorithm
    start_time = time.time()
    best_individuals_data = diff_evo_alg(S_pop, config, start_time)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'''
    Function: {config.function.func_type.name}
    Num of arguments: {config.nr_of_args}
    Execution time: {round(execution_time,2)} seconds
    ''')

    print(best_individuals_data[0][1].fitness_value)
    print(best_individuals_data[-1][1].fitness_value)

    # Connect to database
    db_config = DatabaseConfig()
    conn = SQLiteConnector(db_config.database)
    conn.connect()

    # Creating table
    table_name = get_table_name(config)
    table_name = conn.create_table(table_name)

    # Inserting data into database
    formatted_best_individuals = format_individuals(best_individuals_data)
    conn.insert_multiple_best_individuals(table_name, formatted_best_individuals)

    conn.close()


if __name__ == "__main__":
    # global optimum
    # pprint.pprint(test_fun(FUNCTIONS.F12014, 20))

    # plot_fun_formula(FUNCTIONS.F1.value)
    # plot_fun(operator.elliptic_func, [(-100, 100), (-100, 100)], 1)

    # for fun in [FUNCTIONS.F19]:
    #     plot_fun(fun, [(-100, 100), (-100, 100)], 2)
    #     # plot_fun(fun, [(-70, -45), (-50, -25)], 2)
    #     print(f"Function Complete {fun}")
    # exit(1)

    for i in range(1):
        config = Config()

        if config.run_all_functions and config.run_all_args:
            for member_dim in config.nr_of_args_arr:
                config.nr_of_args = member_dim
                for enum_fun in FUNCTIONS:
                    fun = FunctionObject(enum_fun, config.nr_of_args)
                    run_de(config, fun)
        elif config.run_all_functions:
            for enum_fun in FUNCTIONS:
                fun = FunctionObject(enum_fun, config.nr_of_args)
                run_de(config, fun)
        else:
            run_de(config)
