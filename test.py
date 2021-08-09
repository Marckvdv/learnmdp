import mdp
import models
import sympy
import numpy as np
import time

models = [models.get_random_deterministic(state_count=100, observation_count=10, input_count=5) for _ in range(500)]
config1 = { 'name': 'linear', 'linear_close': True, 'linear_hypothesis': True, 'tries': 50000, 'max_observation_length': 14 }
config2 = { 'name': 'nolinear', 'linear_close': False, 'linear_hypothesis': False, 'tries': 50000, 'max_observation_length': 14 }

configs = [config1, config2]

for m in models:
	hypotheses = []
	for c in configs:
		table = mdp.ObservationsTable(m, m.observation_mapping, c)
		h = table.learn_mdp2()

		hypotheses.append(h)
		print(f'config "{c["name"]}": {len(h.states)} states')
	print()


