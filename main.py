import mdp
import mdps
import sympy
import numpy as np
import time

start = time.time()
m = mdps.get_simple2n(15)
#m = mdps.get_float_n(3)
print(m.to_dot())
print(f"MDP has {len(m.states)} states")

prefixes = [(m.initial_state.observation, ())]
suffixes = [((), i) for i in m.inputs]
table = mdp.ObservationsTable(prefixes, suffixes, m, m.observation_mapping, False, True)

#v = np.array([0.25,0.75])
#vs = [np.array([0.5,0.5]), np.array([1,0]), np.array([0,1]), v]
#print(table.get_vector_decompositions(vs))
h = table.learn_mdp2()
#table.print_observation_table()
print(h.to_dot())
#h = table.create_hypothesis()
#cex = m.try_find_counter_example(h)
end = time.time()
print("took {}ms".format((end-start)*1000))
print(table.stats)
