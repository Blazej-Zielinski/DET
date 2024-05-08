from diffEvoLib.diffEvoAlgs.base import BaseDiffEvoAlg
from diffEvoLib.diffEvoAlgs.data.alg_data import DELBData


class DELB(BaseDiffEvoAlg):
    def __init__(self, params: DELBData, db_conn=None, db_auto_write=False):
        super().__init__(DELB.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr

    def next_epoch(self):
        pass
