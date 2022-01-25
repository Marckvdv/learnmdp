import mdp
import observation_table
import models
#import coffee_machine
#import coin2
#import first
#import second
#import slot_machine

import sympy
import numpy as np
import time
import sys

import random
#random.seed(0)

start = time.time()
#m = models.get_chain(int(sys.argv[1]))
#m = models.get_float_n(3)
#m = models.get_test1()
#m = coin2.coin2
m = models.get_simple3()

#print("coin", coin2.coin2.get_max_depth(), len(coin2.coin2.states))
#print("coffee_machine", coffee_machine.coffee_machine.get_max_depth(), len(coffee_machine.coffee_machine.states))
#print("first", first.first.get_max_depth(), len(first.first.states))
#print("second", second.second.get_max_depth(), len(second.second.states))
#print("slot_machine", slot_machine.slot_machine.get_max_depth(), len(slot_machine.slot_machine.states))
#print(m.to_dot())
#print(f"MDP has {len(m.states)} states")
#assert m.check()
config1 = { 'linear': True, 'tries': 5000, 'max_observation_length': 14, 'cex': 'all_suffixes'}
#
table = observation_table.ObservationTable(m, m.observation_mapping, config1)

h = table.learn_mdp()
#assert m.try_find_counter_example(h, 100, 40) is None
table.print_observation_table()
table.print_observation_table_latex()
print(h.to_dot())
print(f"learned mdp has {len(h.states)} states")
#h = table.create_hypothesis()
#cex = m.try_find_counter_example(h)
end = time.time()
print("took {}ms".format((end-start)*1000))
print(table.stats)
