import copy
import random
import numpy as np
from math import exp

PC = 0
FC = 0
d11 = 1 - (1 - PC) * exp(-PC)
d12 = 1 - (1 - FC) * exp(-FC)

d21 = 2 * (1 - (1 + PC)) * exp(-PC)
d22 = 2 * (1 - (1 + FC)) * exp(-FC)
