import mdp
import observation_table
import models

import sympy
import numpy as np
import time

start = time.time()
#m = models.get_chain(3)
#m = models.get_float_n(3)
m = models.get_test1()
print(m.to_dot())
print(f"MDP has {len(m.states)} states")
assert m.check()
config1 = { 'linear_close': True, 'linear_hypothesis': True, 'tries': 100, 'max_observation_length': 5, 'cex': 'all_suffixes'}

table = observation_table.ObservationTable(m, m.observation_mapping, config1)

h = table.learn_mdp()
assert m.try_find_counter_example(h, 100, 15) is None
table.print_observation_table()
print(h.to_dot())
print(f"learned mdp has {len(h.states)} states")
#h = table.create_hypothesis()
#cex = m.try_find_counter_example(h)
end = time.time()
print("took {}ms".format((end-start)*1000))
print(table.stats)
