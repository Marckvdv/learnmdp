import mdp
import models
import sympy
import numpy as np
import time
import itertools
import sys
import matplotlib
from multiprocessing import Process

output_file = sys.argv[1] if len(sys.argv) > 1 else 'bench_results'

def do_benchmark(model, config, n=3):
	assert model.check()
	output_name = f'{output_file}_{model.name}_{"linear" if config["linear_close"] else "non_linear"}.txt'
	with open(output_name, 'w') as f:
		for _ in range (n):
			start = time.time()

			table = mdp.ObservationsTable(model, model.observation_mapping, config)

			h = table.learn_mdp()
			end = time.time()
			delta_time = end - start

			f.write(f"{model.name} ({len(model.states)} states) {config}:\n")
			f.write(f"{len(h.states)} states\n")
			f.write(f"{delta_time} seconds\n")
			f.write(f"{table.stats}\n")
			f.write("\n")
			f.flush()

def main():
	print(f"writing output to prefix '{output_file}' is that okay?")
	input()

	config1 = { 'linear_close': True, 'linear_hypothesis': True, 'tries': 50000, 'max_observation_length': 14 }
	config2 = { 'linear_close': False, 'linear_hypothesis': False, 'tries': 50000, 'max_observation_length': 14 }

	#configs = [config1, config2]
	configs = [config1]
	#configs = [config2]
	m1 = [models.get_simple2n(n) for n in range(1, 12)]
	m2 = [models.get_chain(n) for n in range(1, 12)]
	models = []
	model_n = {}

	for m_a, m_b in zip(m1, m2):
		models.append(m_a)
		models.append(m_b)

	processes = []
	for model, config in itertools.product(models, configs):
		p = Process(target=do_benchmark, args=(model, config, 3))
		p.start()
		p.join()

if __name__ == '__main__':
	main()
