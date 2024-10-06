from src.enums.strategies import StrategiesEnum


class Strategy:
    def __init__(self, strategy: StrategiesEnum, probability: float):
        self.strategy = strategy
        self.probability = probability
        self.ns = 0
        self.nf = 0

    def set_probability(self, probability: float):
        self.probability = probability

    def get_probability(self):
        return self.probability

    def set_ns(self, ns: int):
        self.ns = ns

    def set_nf(self, nf: int):
        self.nf = nf

    def update_ns(self):
        self.ns += 1

    def update_nf(self):
        self.nf += 1

    def reset_nf_ns(self):
        self.ns = 0
        self.nf = 0
