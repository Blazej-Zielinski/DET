from dataclasses import dataclass

from diffEvoLib.models.fitness_function import FitnessFunctionBase
from diffEvoLib.models.enums.boundary_constrain import BoundaryFixing
from diffEvoLib.models.enums.optimization import OptimizationType


@dataclass
class BaseData:
    num_of_epochs: int
    population_size: int
    nr_of_args: int
    interval_lower_bound: float
    interval_higher_bound: float
    mode: OptimizationType
    boundary_constraints_fun: BoundaryFixing
    function: FitnessFunctionBase


@dataclass
class DefaultAlgData(BaseData):
    mutation_factor: float
    crossover_rate: float


@dataclass
class BestWorstData(BaseData):
    mutation_factor: float
    crossover_rate: float


@dataclass
class RandomLocationsData(BaseData):
    mutation_factor: float
    crossover_rate: float


@dataclass
class NovelModifiedData(BaseData):
    delta_f: float
    delta_cr: float
    sp: int


@dataclass
class AdaptiveParamsData(BaseData):
    prob_f: float
    prob_cr: float


@dataclass
class EmDeData(BaseData):
    crossover_rate: float


@dataclass
class ScalingParamsData(BaseData):
    pass


@dataclass
class DELBData(BaseData):
    crossover_rate: float
    w_factor: float  # control frequency of local exploration around trial and best vectors


@dataclass
class OppBasedData(BaseData):
    mutation_factor: float
    crossover_rate: float
    max_nfc: float
    jumping_rate: float
    threshold: float


@dataclass
class DEGLData(BaseData):
    mutation_factor: float
    crossover_rate: float
    radius: int  # neighborhood size, 2k + 1 <= NP, at least k=2
    weight: float  # controls the balance between the exploration and exploitation


@dataclass
class JADEData(BaseData):
    archive_size: int

    mutation_factor_mean: float
    mutation_factor_std: float

    crossover_rate_mean: float
    crossover_rate_std: float
    crossover_rate_low: float
    crossover_rate_high: float

    c: float  # describes the rate of parameter adaptation
    p: float  # describes the greediness of the mutation strategy


@dataclass
class AADEData(BaseData):
    mutation_factor: float
    crossover_rate: float
