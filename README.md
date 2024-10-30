# DetPy (Differential Evolution Tools): A Python toolbox for solving optimization problems using differential evolution

# Introduction
The DetPy library contains implementations of the differential evolution algorithm and 15 modifications of this 
algorithm. It can be used to solve advanced optimization problems.
The following variants have been implemented:



# Installation
```
pip install detpy
```

# Using FunctionLoader

You can also use one of predefined functions to solve your problem. 
To do this, call the FunctionLoader method and pass as an argument the name of a function from the folder and variables,
which u want to use in your calculations.

```
function_loader = FunctionLoader()
function_name = "ackley"
variables = [0.0, 0.0]
n_dimensions = 2

result = function_loader.evaluate_function(function_name, variables, n_dimensions)
```

Available functions:

```
        self.function_classes = {
            "ackley": Ackley,
            "rastrigin": Rastrigin,
            "rosenbrock": Rosenbrock,
            "sphere": Sphere,
            "griewank": Griewank,
            "schwefel": Schwefel,
            "michalewicz": Michalewicz,
            "easom": Easom,
            "himmelblau": Himmelblau,
            "keane": Keane,
            "rana": Rana,
            "pits_and_holes": PitsAndHoles,
            "hypersphere": Hypersphere,
            "hyperellipsoid": Hyperellipsoid,
            "eggholder": EggHolder,
            "styblinski_tang": StyblinskiTang,
            "goldstein_and_price": GoldsteinAndPrice
        }
```

Test functions prepared based on https://gitlab.com/luca.baronti/python_benchmark_functions

# References