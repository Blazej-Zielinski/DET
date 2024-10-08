import json


class Result:
    def __init__(self, params):
        self.epoch = params[1]
        self.arguments_best = json.loads(params[2])
        self.fitness_value_best = params[3]
        self.arguments_worst = json.loads(params[4])
        self.fitness_value_worst = params[5]
        self.mean = params[6]
        self.std = params[7]
        self.exec_time = params[8]

    def __str__(self):
        return f'''
            Result: [
            \t Epoch: {self.epoch}
            \t Arguments: [{''.join(str(argument) + '; ' for argument in self.arguments_best)}]
            \t Fitness value: {self.fitness_value_best}
            \t Mean: {self.mean}
            \t Std: {self.std}
            \t Execution time: {round(self.exec_time,2)}
            ]        
        '''

    def __repr__(self):
        return self.__str__()
