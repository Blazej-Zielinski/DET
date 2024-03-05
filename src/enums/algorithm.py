from enum import Enum
from src.algorithms.algs import default_alg, best_worst_alg, random_locations_alg, novel_modified_de, pcx_de, pbx_de, \
    laplace_de, tensegrity_structures_de, bidirectional_de, adaptive_params_de, em_de, scaling_params_de, \
    self_adaptive_de, jade
from src.algorithms.initializers import nm_initialize, pbx_initialize, adaptive_params_initialize, sa_initialize, jade_initialize


class AlgorithmType(Enum):
    # good
    DEFAULT = 'default'
    BEST_WORST = 'best_worst'
    RANDOM_LOCATION = 'random_location'
    EM_DE = 'em_de'

    ADAPTIVE_PARAMS = 'adaptive_params'
    NOVEL_MODIFIED = 'novel_modified'
    SCALING_PARAMS = 'scaling_params'

    # mid
    LAPLACE = 'laplace'

    # bad
    BIDIRECTIONAL = 'bidirectional'
    PARENT_CENTRIC = 'parent_centric'
    PBX = 'pbx'
    TENSEGRITY = 'tensegrity'

    # new
    SELF_ADAPTIVE = 'self_adaptive'
    GDE3 = 'gde3'
    FUZZY = 'fuzzy'
    JADE = 'jade'
    OPPOSITION_BASED = 'opposition_based'


def get_algorithm(alg_type: AlgorithmType):
    return {
        AlgorithmType.DEFAULT: (lambda pop, config, curr_gen, args: default_alg(pop, config), None),
        AlgorithmType.BEST_WORST: (lambda pop, config, curr_gen, args: best_worst_alg(pop, config, curr_gen), None),
        AlgorithmType.RANDOM_LOCATION: (lambda pop, config, curr_gen, args: random_locations_alg(pop, config), None),
        AlgorithmType.NOVEL_MODIFIED: (
            lambda pop, config, _, args: novel_modified_de(pop, config, args),
            lambda config: nm_initialize(config)
        ),
        AlgorithmType.PARENT_CENTRIC: (lambda pop, config, curr_gen, args: pcx_de(pop, config), None),
        AlgorithmType.PBX: (
            lambda pop, config, curr_gen, args: pbx_de(pop, config, curr_gen + 1, args),
            lambda config: pbx_initialize()
        ),
        AlgorithmType.LAPLACE: (lambda pop, config, curr_gen, args: laplace_de(pop, config), None),
        AlgorithmType.TENSEGRITY: (lambda pop, config, curr_gen, args: tensegrity_structures_de(pop, config), None),
        AlgorithmType.BIDIRECTIONAL: (lambda pop, config, curr_gen, args: bidirectional_de(pop, config), None),
        AlgorithmType.ADAPTIVE_PARAMS: (
            lambda pop, config, curr_gen, args: adaptive_params_de(pop, config, args),
            lambda config: adaptive_params_initialize(config)
        ),
        AlgorithmType.EM_DE: (lambda pop, config, curr_gen, args: em_de(pop, config), None),
        AlgorithmType.SCALING_PARAMS: (
            lambda pop, config, curr_gen, args: scaling_params_de(pop, config, curr_gen + 1), None),
        AlgorithmType.SELF_ADAPTIVE: (
            lambda pop, config, curr_gen, args: self_adaptive_de(pop, config, curr_gen, args),
            lambda config: sa_initialize(config)),
        AlgorithmType.JADE: (
            lambda pop, config, curr_gen, args: jade(pop, config, args),
            lambda config: jade_initialize(config)
        )
    }.get(alg_type, lambda: None)
