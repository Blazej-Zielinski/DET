

class Optimum:
    def __init__(self, params):
        self.algorithm = params[1]
        self.function = params[2]
        self.optimum_found = params[3]
        self.nr_of_successes = params[4]
        self.avg_epoch_optimum = params[5]

    def __str__(self):
        return f'''
            Optimum: [
            \t Algorithm: {self.algorithm}
            \t Function: {self.function}
            \t Optimum found: {bool(self.optimum_found)}
            \t Nr of successes: {self.nr_of_successes}
            \t Avg epoch optimum: {self.avg_epoch_optimum}
            ]        
        '''

    def __repr__(self):
        return self.__str__()
