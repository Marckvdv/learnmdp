import random
import numpy as np
from scipy import optimize
import itertools
import typing
import util
import math

class State:
	def __init__(self, name: str, observation: int, transitions: dict = {}):
		self.name = name if name else ''
		self.observation = observation
		self.transitions = transitions if transitions else {}

	def add_transition(self, input_symbol, next_state, probability):
		#print(f"adding transition {self} {input_symbol} {next_state} {probability}")
		if input_symbol not in self.transitions:
			self.transitions[input_symbol] = {}
		self.transitions[input_symbol][next_state] = probability
		#print("AFTER:", self.transitions)

	def check(self):
		for i, v in self.transitions.items():
			total = sum(s for s in v.values())
			if not math.isclose(total, 1.0):
				print(self.transitions)
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
		return '#' + self.name

class MDP:
	def __init__(self, states, initial_state, inputs, observations):
		assert initial_state in states

		self.states = states
		self.initial_state = initial_state
		self.inputs = inputs
		self.observations = observations
		self.observation_mapping = { i:n for n, i in enumerate(observations) }

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
				for s2, p2 in s1.transitions[current_input].items():
					if s2.observation == current_observation:
						output_prob += p1*p2

			for s1, p1 in current_state_distribution.items():
				for s2, p2 in s1.transitions[current_input].items():
					if s2.observation == current_observation:
						p = p1*p2/output_prob
						if s2 in new_state_distribution: new_state_distribution[s2] += p
						else: new_state_distribution[s2] = p

			current_trace = current_trace[1:]
			current_state_distribution = new_state_distribution

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
	
	def try_find_counter_example(self, other, tries=1000, max_observation_length=4):
		for _ in range(tries):
			prefix, final_input = self.random_trace(max_observation_length)
			d1 = self.get_exact_output_distribution_io(prefix, final_input)
			d2 = other.get_exact_output_distribution_io(prefix, final_input)
			if not util.distribution_eq(d1, d2):
				return prefix, final_input
		return None

	def try_find_short_counter_example(self, other, tries=1000, max_observation_length=4):
		print(other.to_dot())
		# TODO optimize
		for _ in range(tries):
			for n in range(max_observation_length):
				prefix, final_input = self.random_trace(n)

				d1 = self.get_exact_output_distribution_io(prefix, final_input)
				d2 = other.get_exact_output_distribution_io(prefix, final_input)

				print(prefix, final_input, ":", d1, "vs", d2)
				if not util.distribution_eq(d1, d2):
					#print("PREFIX:", prefix)
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
						s += f'{state.name} -> {next_state.name} [label="{input_symbol}:{probability}" ];\n'

		for state in self.states:
			s += f'{state.name}[label="{state.observation}"]\n'

		s += "}"
		return s

class ObservationsTable:
	def __init__(self, prefixes, suffixes, mdp, observation_mapping, linear_close, linear_hypothesis):
		self.short_prefixes = set(prefixes)
		self.suffixes = set(suffixes)
		self.distributions = {}
		self.mdp = mdp
		self.observation_mapping = observation_mapping
		self.linear_close = linear_close
		self.linear_hypothesis = linear_hypothesis
		self.stats = { 'membership_queries': 0, 'equivalence_queries': 0, 'make_consistent': 0, 'make_closed': 0}

		self.add_long_prefixes()
		self.fill_table()

	def make_closed(self):
		self.stats['make_closed'] += 1
		if self.linear_close:
			return self.make_closed_linear()
		else:
			return self.make_closed_non_linear()
	
	def make_closed_non_linear(self):
		for l in self.long_prefixes:
			all_different = True
			for s in self.short_prefixes:
				if self.row_eq(l, s):
					all_different = False
					break

			if all_different:
				self.short_prefixes.add(l)
				self.long_prefixes.remove(l)
				return True
		return False

#	def make_closed_linear(self):
#		for l in self.long_prefixes:
#			# Check if l is simply present in short prefixes
#			all_different = True
#			for s in self.short_prefixes:
#				if self.row_eq(l, s):
#					all_different = False
#					break
#
#			if all_different:
#				vector = np.hstack([self.get_matrix_row(l), [1]])
#				matrix = np.array([self.get_matrix_row(ll) for ll in (self.long_prefixes|self.short_prefixes) if ll != l]).T
#				vector_count = matrix.shape[1]
#				matrix = np.vstack([matrix, np.ones(vector_count)])
#				solution = optimize.linprog(np.zeros(vector_count), A_eq=matrix, b_eq=vector)
#
#				if not solution.success:
#					self.short_prefixes.add(l)
#					self.long_prefixes.remove(l)
#					return True
#		return False

	def make_closed_linear(self):
		#self.print_observation_table()
		short_vectors = [self.get_matrix_row(v) for v in self.short_prefixes]
		independent_vectors = []
		row_to_prefix = {}
		for l in self.long_prefixes:
			vector = self.get_matrix_row(l)
			if any(map(lambda v: np.array_equal(v, vector), independent_vectors)):
				continue

			row_to_prefix[util.HashableArray(vector)] = l

			solution = self.get_probabilistic_decomposition(short_vectors, vector)
			if solution is None:
				independent_vectors.append(vector)

		decomps = self.get_vector_decompositions(independent_vectors)
		change = False
		for i, d in zip(independent_vectors, decomps):
			if d is None:
				change = True
				prefix = row_to_prefix[util.HashableArray(i)]
				self.short_prefixes.add(prefix)
				self.long_prefixes.remove(prefix)
		return change

	def make_consistent(self):
		self.stats['make_consistent'] += 1

		for s1, s2 in itertools.combinations(self.short_prefixes, 2):
			if not self.row_eq(s1, s2):
				continue

			for i, o in itertools.product(self.mdp.inputs, self.mdp.observations):
				t = self.distributions[(s1, i)]
				#print("T:",t, "o:", o)
				if t and o in t and t[o] > 0 and not self.row_eq((s1[0], s1[1] + ((i, o),)), (s2[0], s2[1] + ((i, o),))):
					for e in self.suffixes:
						t1 = self.distributions[((s1[0], s1[1] + ((i, o),) + e[0]), e[1])]
						t2 = self.distributions[((s2[0], s2[1] + ((i, o),) + e[0]), e[1])]
						if t1 != t2:
							self.suffixes.add((e[0] + ((i, o),), e[1]))
							return True
		return False

	def add_long_prefixes(self):
		self.long_prefixes = set()
		for p, i, o in itertools.product(self.short_prefixes, self.mdp.inputs, self.mdp.observations):
			new_trace = (p[0], p[1] + ((i, o),))
			if new_trace not in self.short_prefixes and self.get_probability(p, i, o) > 0:
				self.long_prefixes.add(new_trace)
	
	def fill_table(self):
		for p in self.short_prefixes | self.long_prefixes:
			for s in self.suffixes:
				trace = (p[0], p[1] + s[0])
				if (trace, s[1]) not in self.distributions:
					self.stats['membership_queries'] += 1
					t = self.mdp.get_exact_output_distribution_io(trace, s[1])
					self.distributions[(trace, s[1])] = t

	def get_distribution(self, prefix, last_input):
		if (prefix, last_input) not in self.distributions:
			self.stats['membership_queries'] += 1
			self.distributions[(prefix, last_input)] = self.mdp.get_exact_output_distribution_io(prefix, last_input)

		return self.distributions[(prefix, last_input)]
	
	def get_probability(self, prefix, last_input, output):
		t = self.get_distribution(prefix, last_input)
		if output in t:
			return t[output]
		else:
			return 0

	def print_observation_table(self):
		n = 50
		print(" "*n, end='')
		first = True
		for s in self.suffixes:
			if first:
				print('|| ', end='')
				first = False
			print(f'{str(s).ljust(n)}', end='')
		print('\n' + '='*n*(len(self.suffixes)+1))

		for p in self.short_prefixes:
			print(f'{str(p).ljust(n)}|| ', end='')
			for s in self.suffixes:
				trace = p + s[0]
				print(str(self.get_distribution(trace, s[1])).ljust(n), end='')
			print()

		print('-'*n*(len(self.suffixes)+1))

		for p in self.long_prefixes:
			print(f'{str(p).ljust(n)}|| ', end='')
			for s in self.suffixes:
				trace = p + s[0]
				print(str(self.get_distribution(trace, s[1])).ljust(n), end='')
			print()
	
	def get_row(self, prefix):
		#print("PREFIX:", prefix)
		row = []
		for s in self.suffixes:
			row.append(self.get_distribution((prefix[0], prefix[1] + s[0]), s[1]))
		return row
	
	#def row_to_vector(self, row, final_observation):
	#	initial, rest = row

	#	size = len(self.suffixes)*(len(rest)+1)
	#	vector = np.zeros(size)
	#	print("ROW", row)
	#	vector[self.observation_mapping[final_observation]] = 1
	#	for i, r in enumerate(rest):
	#		for o, p in r:
	#			idx = i*len(self.mdp.inputs) + self.observation_mapping[o]
	#			vector[idx] = p
	#	return vector
	
	def get_matrix_row(self, prefix):
		size = (len(self.suffixes)+1)*len(self.mdp.observations)
		#row = sympy.zeros(size)
		vector = np.zeros(size)
		last_observation = self.get_last(prefix)
		vector[self.observation_mapping[last_observation]] = 1

		#print("PREFIX:", prefix)
		for i, s in enumerate(self.suffixes):
			dist = self.get_distribution((prefix[0], prefix[1] + s[0]), s[1])
			for o, p in dist.items():
				idx = (i+1)*len(self.mdp.observation_mapping) + self.observation_mapping[o]
				vector[idx] = p

		return vector
	
	def fix_table(self):
		self.add_long_prefixes()
		self.fill_table()
	
	def row_eq(self, p1, p2):
		if self.get_last(p1) != self.get_last(p2):
			return False

		r1, r2 = self.get_row(p1), self.get_row(p2)
		for v1, v2 in zip(r1, r2):
			if not util.distribution_eq(v1, v2): return False
		return True
	
	def learn_mdp(self):
		iteration = 0
		while True:
			print(f"Iteration {iteration}")

			self.fix_table()
			while self.make_closed() or self.make_consistent():
				self.fix_table()

			#self.print_observation_table()

			h = self.create_hypothesis()
			#print(h.to_dot())
			assert h.check()

			self.stats['equivalence_queries'] += 1
			cex = self.mdp.try_find_counter_example(h)
			if cex == None:
				return h

			#(initial_output, current_prefix), final_input = cex
			(initial_output, rest), final_input = cex
			minimal_prefix = self.get_minimal_prefix(cex)
			print("MININMAL PREFIX: ", minimal_prefix)
			self.short_prefixes.add(minimal_prefix)
			#prefix = initial_output, tuple(rest)
			#print("Adding: ", prefix)
			#print("TO:", self.short_prefixes)

			iteration += 1

	def learn_mdp1(self):
		self.print_observation_table()
		iteration = 0
		while True:
			print(f"Iteration {iteration}")

			self.fix_table()
			while self.make_closed() or self.make_consistent():
				self.fix_table()

			#self.print_observation_table()

			h = self.create_hypothesis()
			#print(h.to_dot())
			assert h.check()

			self.stats['equivalence_queries'] += 1
			cex = self.mdp.try_find_counter_example(h)
			if cex == None:
				return h

			(initial_output, current_prefix), final_input = cex
			while current_prefix:
				p = initial_output, tuple(current_prefix)
				self.short_prefixes.add(p)
				self.long_prefixes.discard(p)
				current_prefix = current_prefix[:-1]

			iteration += 1

	def learn_mdp2(self):
		iteration = 0
		while True:
			print(f"Iteration {iteration}")

			self.fix_table()
			while self.make_closed() or self.make_consistent():
				self.fix_table()

			h = self.create_hypothesis()
			assert h.check()

			self.stats['equivalence_queries'] += 1
			cex = self.mdp.try_find_counter_example(h)
			if cex == None:
				return h

			#(initial_output, current_prefix), final_input = cex
			(initial_output, current_suffix), final_input = cex
			while current_suffix:
				p = tuple(current_suffix), final_input
				#print(f"Adding suffix {p}")
				self.suffixes.add(p)
				current_suffix = current_suffix[1:]
			self.print_observation_table()

			iteration += 1
	
	def get_minimal_prefix(self, cex):
		current_prefix = cex
		(initial, current_rest), current_final = cex
		while current_rest:
			next_rest, next_final = current_rest[:-1], current_rest[-1][0]
			prefix = initial, tuple(next_rest)

			if prefix in self.short_prefixes:
				return initial, tuple(current_rest)

			current_rest, current_final = next_rest, next_final
		assert False
		return initial, ()
	
	def get_last(self, prefix):
		if prefix[1]:
			return prefix[1][-1][1]
		else:
			return prefix[0]

	def create_hypothesis(self):
		if self.linear_hypothesis:
			return self.create_hypothesis_linear()
		else:
			return self.create_hypothesis_non_linear()
	
	def create_hypothesis_non_linear(self):
		states = {}
		prefix_to_row = {}
		
		sorted_prefixes = sorted(self.short_prefixes, key=lambda k: len(k[1]))
		for p in sorted_prefixes:
			output = self.get_last(p)
			row = output, tuple(frozenset(v.items()) for v in self.get_row(p))
			if row not in states:
				states[row] = State(f's{len(states)}', output)
			prefix_to_row[p] = row

		for p in self.long_prefixes:
			output = self.get_last(p)
			row = output, tuple(frozenset(v.items()) for v in self.get_row(p))
			prefix_to_row[p] = row

		initial_prefix = self.mdp.initial_state.observation, ()
		initial_state = states[prefix_to_row[initial_prefix]]

		#print("STATES", states)
		for p in self.short_prefixes:
			row = prefix_to_row[p]
			state = states[row]
			#print("STATE:", state)

			for i, o in itertools.product(self.mdp.inputs, self.mdp.observations):
				extended_trace = p[0], p[1] + ((i, o),)
				prob = self.get_probability(p, i, o)
				if prob == 0:
					continue
				#print("\tIO:", i, o)
				#print("\tPROB", prob)

				next_row = prefix_to_row[extended_trace]
				#print("NR:", next_row)
				#print("STS:", states)
				next_state = states[next_row]
				state.add_transition(i, next_state, prob)

		return MDP(list(states.values()), initial_state, self.mdp.inputs, self.mdp.observations)

	def create_hypothesis_linear(self):
		states = {}
		prefix_to_row = {}

		sorted_prefixes = sorted(self.short_prefixes, key=lambda k: len(k[1]))
		for p in sorted_prefixes:
			row = self.get_matrix_row(p)
			if util.HashableArray(row) not in states:
				output = self.get_last(p)
				states[util.HashableArray(row)] = State(f's{len(states)}', output)
			prefix_to_row[p] = row

		rows = list(map(lambda p: self.get_matrix_row(p), prefix_to_row.keys()))

		# TODO optimize
		#unique_rows = []
		#for r in rows:
		#	if not any(map(lambda v: np.array_equal(v, r), unique_rows)):
		#		unique_rows.append(r)

		#vector_decomposition = self.get_vector_decompositions(unique_rows)
		#prime_rows = []
		#for r, d in zip(unique_rows, vector_decomposition):
		#	if d is None:
		#		prime_rows.append(r)
		#print("PRIME ROWS:", prime_rows)
		#input()
		prime_rows = rows

		for p in self.long_prefixes:
			prefix_to_row[p] = row

		initial_prefix = self.mdp.initial_state.observation, ()
		initial_state = states[util.HashableArray(prefix_to_row[initial_prefix])]

		rows_done = set()
		for p, r in prefix_to_row.items():
			if util.HashableArray(r) in rows_done or not any(map(lambda v: np.array_equal(v, r), prime_rows)):
				continue
			rows_done.add(util.HashableArray(r))

			for i, o in itertools.product(self.mdp.inputs, self.mdp.observations):
				extended_trace = p[0], p[1] + ((i, o),)
				prob = self.get_probability(p, i, o)
				if prob == 0 or extended_trace not in prefix_to_row:
					continue
				next_row = prefix_to_row[extended_trace]
				# Calculate probabilistic decomposition again

				#if next_row in states:
				#if any(map(lambda v: np.array_equal(v, next_row), prime_rows)):
				#if next_row in prime_rows:
				#	next_state = states[util.HashableArray(next_row)]
				#	states[util.HashableArray(r)].add_transition(i, next_state, prob)
				#else:
				vector = self.get_matrix_row(extended_trace)
				solution = self.get_probabilistic_decomposition(prime_rows, vector)
				assert solution is not None

				for row, prob in solution.items():
					#print("ROW:", row)
					states[util.HashableArray(r)].add_transition(i, states[row], prob)


		return MDP(list(states.values()), initial_state, self.mdp.inputs, self.mdp.observations)
	
	def get_probabilistic_decomposition(self, vectors, vector):
		# Get unique vectors and establish an ordering
		unique_vectors = list(set([util.HashableArray(v) for v in vectors]))
		vector = np.hstack([vector, [1]])
		matrix = np.array([v.arr for v in unique_vectors]).T
		vector_count = matrix.shape[1]
		matrix = np.vstack([matrix, np.ones(vector_count)])
		solution = optimize.linprog(np.zeros(vector_count), A_eq=matrix, b_eq=vector, method='simplex')

		if solution.success:
			probs = { unique_vectors[i]: solution.x[i] for i in range(len(unique_vectors)) }
			return probs
		else:
			return None
	
	def get_vector_decompositions(self, vectors):
		if len(vectors) == 0: return []
		elif len(vectors) == 1: return [None]

		# We assume all vectors to be unique
		vs = vectors
		solutions = []
		for i in range(len(vectors)):
			v = vs.pop(0)
			solution = self.get_probabilistic_decomposition(vs, v)
			solutions.append(solution)
			vs.append(v)
		return solutions

if __name__ == '__main__':
	print("Wrong file!")
