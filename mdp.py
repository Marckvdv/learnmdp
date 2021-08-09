import random
import numpy as np
import itertools
import typing
import util
import math
from pprint import pprint

class State:
	def __init__(self, name: str, observation: int, transitions: dict = {}):
		self.name = name if name else ''
		self.observation = observation
		self.transitions = transitions if transitions else {}

	def add_transition(self, input_symbol, next_state, probability):
		if input_symbol not in self.transitions:
			self.transitions[input_symbol] = {}
		assert next_state not in self.transitions[input_symbol]
		self.transitions[input_symbol][next_state] = probability

	def check(self):
		for i, v in self.transitions.items():
			total = sum(s for s in v.values())
			if not math.isclose(total, 1.0):
				return False
		return True

	def sample_next_state(self, input_symbol):
		r = random.random()
		t = 0
		for s, p in self.transitions[input_symbol].items():
			t += p
			if r <= t: return s
		assert False
	
	def __repr__(self):
		return f'#{self.name}:{self.observation}'

class MDP:
	def __init__(self, states, initial_state, inputs, observations, name='mdp'):
		assert initial_state in states

		self.states = states
		self.initial_state = initial_state
		self.inputs = inputs
		self.observations = observations
		self.observation_mapping = { i:n for n, i in enumerate(observations) }
		self.name = name

	def check(self):
		return all(s.check() for s in self.states)

	def get_exact_state_distribution_io(self, io_trace, final_input):
		if io_trace[0] != self.initial_state.observation:
			return {}

		current_state_distribution = {self.initial_state: 1}
		current_trace = io_trace[1]
		while current_trace:
			current_input, current_observation = current_trace[0]
			new_state_distribution = {}
			output_prob = 0
			for s1, p1 in current_state_distribution.items():
				#if current_input not in s1.transitions:
				#	continue
				for s2, p2 in s1.transitions[current_input].items():
					if s2.observation == current_observation:
						output_prob += p1*p2

			if output_prob == 0:
				return {}

			for s1, p1 in current_state_distribution.items():
				for s2, p2 in s1.transitions[current_input].items():
					if s2.observation == current_observation:
						p = p1*p2/output_prob
						if s2 in new_state_distribution: new_state_distribution[s2] += p
						else: new_state_distribution[s2] = p

			current_trace = current_trace[1:]
			current_state_distribution = new_state_distribution
			assert math.isclose(sum(current_state_distribution.values()), 1)

		final_distribution = {}
		for s1, p1 in current_state_distribution.items():
			for s2, p2 in s1.transitions[final_input].items():
				p = p1*p2
				if s2 in final_distribution: final_distribution[s2] += p
				else: final_distribution[s2] = p

		return final_distribution

	def get_exact_output_distribution_io(self, io_trace, final_input):
		state_distribution = self.get_exact_state_distribution_io(io_trace, final_input)
		output_distribution = {}
		for s, p in state_distribution.items():
			if s.observation not in output_distribution:
				output_distribution[s.observation] = 0
			output_distribution[s.observation] += p

		return output_distribution
	
	def next_distribution(self, current_state_distribution, current_input):
		new_state_distribution = {}	
		for s1, p1 in current_state_distribution.items():
			for s2, p2 in s1.transitions[current_input].items():
				p = p1*p2
				if s2 in new_state_distribution: new_state_distribution[s2] += p
				else: new_state_distribution[s2] = p
		return new_state_distribution
	
	def random_trace(self, length):
		current_state = self.initial_state
		current_trace = []
		for _ in range(length):
			current_input = random.choice(self.inputs)
			current_state = current_state.sample_next_state(current_input)
			current_trace.append( (current_input, current_state.observation) )

		return (self.initial_state.observation, current_trace), random.choice(self.inputs)
	
	@util.start_finish_print
	def try_find_counter_example(self, other, tries=10000, max_observation_length=10):
		for _ in range(tries):
			prefix, final_input = self.random_trace(max_observation_length)
			d1 = self.get_exact_output_distribution_io(prefix, final_input)
			d2 = other.get_exact_output_distribution_io(prefix, final_input)
			if not util.distribution_eq(d1, d2):
				#print(f'for input {prefix}:{final_input}, expected {d1} but got {d2}')
				return prefix, final_input
		return None

	def to_dot(self):
		s = "digraph {\n"

		s += "init[style=invis];\n"
		s += f"init -> {self.initial_state.name};\n"

		for state in self.states:
			for input_symbol, mapping in state.transitions.items():
				for next_state, probability in mapping.items():
					if probability != 0:
						s += f'{state.name} -> {next_state.name} [label="{input_symbol}:{round(float(probability), 2)}" ];\n'

		for state in self.states:
			s += f'{state.name}[label="{state.observation}"]\n'

		s += "}"
		return s

if __name__ == '__main__':
	print("Wrong file!")
