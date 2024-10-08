from dataclasses import dataclass, field
from typing import Optional

from DET.models.fitness_function import FitnessFunctionBase
from DET.models.enums.boundary_constrain import BoundaryFixing
from DET.models.enums.optimization import OptimizationType


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
    log_population: bool
    optimum: Optional[float] = field(init=False, default=None)
    tolerance: Optional[float] = field(init=False, default=None)
    parallel_processing: Optional[list] = field(init=False, default=None)


@dataclass
class DEData(BaseData):
    mutation_factor: float
    crossover_rate: float


@dataclass
class COMDEData(BaseData):
    mutation_factor: float
    crossover_rate: float


@dataclass
class DERLData(BaseData):
    mutation_factor: float
    crossover_rate: float


@dataclass
class NMDEData(BaseData):
    delta_f: float
    delta_cr: float
    sp: int


@dataclass
class SADEData(BaseData):
    prob_f: float
    prob_cr: float


@dataclass
class EMDEData(BaseData):
    crossover_rate: float


@dataclass
class IDEData(BaseData):
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


@dataclass
class EIDEData(BaseData):
    crossover_rate_min: float
    crossover_rate_max: float


@dataclass
class MGDEData(BaseData):
    crossover_rate: float
    mutation_factor_f: float
    mutation_factor_k: float
    threshold: float
    mu: float
