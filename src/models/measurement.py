class Measurement:
    def __init__(self, params):
        self.algorithm = params[1]
        self.function = params[2]
        self.epoch = params[3]
        self.avg_fitness_value_best = params[4]

    def __str__(self):
        return f'''
            Measurement: [
            \t Algorithm: {self.algorithm}
            \t Function: {self.function}
            \t Epoch: {self.epoch}
            \t Fitness Value: {self.avg_fitness_value_best}
            ]        
        '''

    def __repr__(self):
        return self.__str__()