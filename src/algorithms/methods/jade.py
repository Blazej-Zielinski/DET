import copy
import random
import numpy as np

from src.models.member import Member

from self_adaptive import sa_mutation_curr_to_best_1

# TODO
# Fi is connected with xi, so every x has its own F
# also F is regenerate in each generation and it is also adapted

# Move strategies from SaDE to somewhere else so they could be used in different algos

# Initializing archive as empty
# In each generation the parent solutions, which fail to success into next gen are added to the archive
# If the size of archive exceeds threshold some of the solutions are randomly removed
