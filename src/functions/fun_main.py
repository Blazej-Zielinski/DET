import opfunu.cec_based.cec2014 as opf
from enum import Enum


class FUNCTIONS(Enum):
    F1 = opf.F12014
    F2 = opf.F22014
    F3 = opf.F32014
    F4 = opf.F42014
    F5 = opf.F52014
    F6 = opf.F62014
    F7 = opf.F72014
    F8 = opf.F82014
    F9 = opf.F92014
    F10 = opf.F102014
    F11 = opf.F112014
    F12 = opf.F122014
    F13 = opf.F132014
    F14 = opf.F142014
    F15 = opf.F152014
    F16 = opf.F162014
    F17 = opf.F172014
    F18 = opf.F182014
    F19 = opf.F192014
    F20 = opf.F202014
    F21 = opf.F212014
    F22 = opf.F222014
    F23 = opf.F232014
    F24 = opf.F242014
    F25 = opf.F252014
    F26 = opf.F262014
    F27 = opf.F272014
    F28 = opf.F282014
    F29 = opf.F292014
    F30 = opf.F302014


class FunctionObject:
    def __init__(self, func_type, ndim):
        self.func_type = func_type
        self.function = func_type.value(ndim=ndim)

    def eval(self, params):
        return self.function.evaluate(params)
