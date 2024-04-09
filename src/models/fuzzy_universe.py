import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class MemberAntecedent:
    def __init__(self, name: str, interval: list[int], mean: float, sigmas: list[float]):
        self.member = ctrl.Antecedent(np.linspace(interval[0], interval[1], 1000 * interval[1]), name)
        self.member['S'] = fuzz.gaussmf(self.member.universe, mean, sigmas[0])
        self.member['M'] = fuzz.gaussmf(self.member.universe, mean, sigmas[1])
        self.member['B'] = fuzz.gaussmf(self.member.universe, mean, sigmas[2])


class MemberConsequent:
    def __init__(self, name: str, interval: list[int], mean: float, sigmas: list[float]):
        self.member = ctrl.Consequent(np.linspace(interval[0], interval[1], 1000 * interval[1]), name)
        self.member['S'] = fuzz.gaussmf(self.member.universe, mean, sigmas[0])
        self.member['M'] = fuzz.gaussmf(self.member.universe, mean, sigmas[1])
        self.member['B'] = fuzz.gaussmf(self.member.universe, mean, sigmas[2])


class RulesSet:
    def __init__(self, first_member: MemberAntecedent, second_member: MemberAntecedent,
                 result_member: MemberConsequent):
        self.first_member = first_member
        self.second_member = second_member
        self.result_member = result_member

        self.rule1 = ctrl.Rule((self.first_member.member['S'] & self.second_member.member['S']),
                               self.result_member.member['S'])
        self.rule2 = ctrl.Rule((self.first_member.member['S'] & self.second_member.member['M']),
                               self.result_member.member['M'])
        self.rule3 = ctrl.Rule((self.first_member.member['S'] & self.second_member.member['B']),
                               self.result_member.member['B'])
        self.rule4 = ctrl.Rule((self.first_member.member['M'] & self.second_member.member['S']),
                               self.result_member.member['S'])
        self.rule5 = ctrl.Rule((self.first_member.member['M'] & self.second_member.member['M']),
                               self.result_member.member['M'])
        self.rule6 = ctrl.Rule((self.first_member.member['M'] & self.second_member.member['B']),
                               self.result_member.member['B'])
        self.rule7 = ctrl.Rule((self.first_member.member['B'] & self.second_member.member['S']),
                               self.result_member.member['B'])
        self.rule8 = ctrl.Rule((self.first_member.member['B'] & self.second_member.member['M']),
                               self.result_member.member['B'])
        self.rule9 = ctrl.Rule((self.first_member.member['B'] & self.second_member.member['B']),
                               self.result_member.member['B'])

    def get_rules(self):
        return [self.rule1, self.rule2, self.rule3, self.rule4, self.rule5, self.rule6, self.rule7, self.rule8,
                self.rule9]


class FuzzySystem:
    # def __init__(self, name: str, interval: list[int], mean: float, sigmas: list[float]):
    def __init__(self, input_params: list[tuple[str, list[int], float, list[float]]],
                 output_params: tuple[str, list[int], float, list[float]]):
        self.first_input_member = MemberAntecedent(input_params[0][0], input_params[0][1], input_params[0][2],
                                                   input_params[0][3])
        self.second_input_member = MemberAntecedent(input_params[1][0], input_params[1][1], input_params[1][2],
                                                    input_params[1][3])

        self.output_member = MemberConsequent(output_params[0], output_params[1], output_params[2], output_params[3])

        self.rules = RulesSet(self.first_input_member, self.second_input_member, self.output_member)

        self.cs = ctrl.ControlSystem(self.rules.get_rules())

        self.css_system = ctrl.ControlSystemSimulation(self.cs)

    def compute_output(self, x: float, y: float) -> float:
        self.css_system.input[self.first_input_member.member.label] = x
        self.css_system.input[self.second_input_member.member.label] = y
        self.css_system.compute()

        return self.css_system.output[self.output_member.member.label]


class FuzzyLogicControl:
    def __init__(self):
        self.f_system = FuzzySystem(input_params=[
            ('D11', [0, 1], 0.25, [0.05, 0.5, 0.9]),

            ('D12', [0, 1], 0.35, [0.01, 0.5, 0.9])
        ], output_params=('F', [0, 2], 0.5, [0.3, 0.6, 0.9]))

        self.cr_system = FuzzySystem(input_params=[
            ('D21', [0, 2], 0.5, [0.1, 0.8, 1.5]),
            ('D22', [0, 2], 0.5, [0.1, 0.8, 1.5])
        ], output_params=('CR', [0, 1], 0.35, [0.1, 0.8, 1.5]))
