import util
import itertools
import numpy as np
from sympy import Rational
import sympy
from scipy import optimize
import mdp
from pprint import pprint

# Table storing all observational data.
class ObservationTable:
	def __init__(self, mdp, observation_mapping, config={}):
		self.short_prefixes = set([(mdp.initial_state.observation, ())])
		self.suffixes = set([((), i) for i in mdp.inputs])
		self.distributions = {}
		self.mdp = mdp
		self.observation_mapping = observation_mapping
		self.stats = {
			'membership_queries': 0,
			'equivalence_queries': 0,
			'make_consistent': 0,
			'make_closed': 0,
		}

		default_config = {
			'linear': True,
			'tries': 1000,
			'max_observation_length': 10,
			'cex': 'all_suffixes',
		}
		self.config = {**default_config, **config}
		self.decomp_cache = {}

		self.add_long_prefixes()
		self.fill_table()

	@util.start_finish_print
	def make_closed(self):
		self.stats['make_closed'] += 1
		if self.config['linear']:
			result = self.make_closed_linear()
		else:
			result = self.make_closed_non_linear()

		return result
	
	def make_closed_non_linear(self):
		for l in self.long_prefixes:
			all_different = True
			for s in self.short_prefixes:
				if (self.get_matrix_row(l) == self.get_matrix_row(s)).all():
					all_different = False
					break

			if all_different:
				self.add_short_prefix(l)
				return True
		return False

	def make_closed_linear(self):
		short_vectors = [self.get_matrix_row(v) for v in self.short_prefixes]
		independent_vectors = []
		row_to_prefix = {}
		for l in self.long_prefixes:
			vector = self.get_matrix_row(l)
			if any(map(lambda v: np.array_equal(v, vector), independent_vectors)):
				continue

			if util.HashableArray(vector) not in row_to_prefix:
				row_to_prefix[util.HashableArray(vector)] = l
			#else:
			#	continue

			solution = self.get_probabilistic_decomposition(short_vectors, vector)
			if solution is None:
				independent_vectors.append(vector)

		decomps = self.get_vector_decompositions(independent_vectors)
		change = False
		for i, d in zip(independent_vectors, decomps):
			if d is None:
				prefix = row_to_prefix[util.HashableArray(i)]
				print("Independent vector", prefix)
				self.add_short_prefix(prefix)
				return True
				change = True
		assert change or len(independent_vectors) == 0

		return change

	#def make_consistent(self):
	#	self.stats['make_consistent'] += 1

	#	for s1, s2 in itertools.combinations(self.short_prefixes, 2):
	#		if not self.row_eq(s1, s2):
	#			continue

	#		for i, o in itertools.product(self.mdp.inputs, self.mdp.observations):
	#			t = self.distributions[(s1, i)]
	#			#print("T:",t, "o:", o)
	#			if t and o in t and t[o] > 0 and not self.row_eq((s1[0], s1[1] + ((i, o),)), (s2[0], s2[1] + ((i, o),))):
	#				for e in self.suffixes:
	#					t1 = self.distributions[((s1[0], s1[1] + ((i, o),) + e[0]), e[1])]
	#					t2 = self.distributions[((s2[0], s2[1] + ((i, o),) + e[0]), e[1])]
	#					if not util.distribution_eq(t1, t2):
	#						self.suffixes.add((e[0] + ((i, o),), e[1]))
	#						return True

	#	return False

	def add_short_prefix(self, p):
		self.short_prefixes.add(p)
		self.long_prefixes.remove(p)
		for i, o in itertools.product(self.mdp.inputs, self.mdp.observations):
			new_trace = (p[0], p[1] + ((i, o),))
			if new_trace not in self.short_prefixes and self.get_probability(p, i, o) > 0:
				self.long_prefixes.add(new_trace)
				for s in self.suffixes:
					trace = (new_trace[0], new_trace[1] + s[0])
					self.get_distribution(trace, s[1])
					#if (trace, s[1]) not in self.distributions:
					#	self.stats['membership_queries'] += 1
					#	t = self.mdp.get_exact_output_distribution_io(trace, s[1])
					#	self.distributions[(trace, s[1])] = t
		

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
				self.get_distribution(trace, s[1])
				#if (trace, s[1]) not in self.distributions:
				#	self.stats['membership_queries'] += 1
				#	t = self.mdp.get_exact_output_distribution_io(trace, s[1])
				#	self.distributions[(trace, s[1])] = t

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

	def print_observation_table_latex(self):
		def prefix_to_string(p):
			initial, rest = p
			s = f"$\\trace{{\\obs{{obs{initial}}}"
			for i, o in rest:
				s += f"\\ {i}\\ \\obs{{obs{o}}}"
			s += "}$"
			return s

		def suffix_to_string(p):
			rest, final = p
			s = "$\\trace{"
			for i, o in rest:
				s += f"{i}\\ \\obs{{obs{o}}}\\ "
			s += final
			s += "}$"
			return s

		def distribution_to_string(d):
			def format_num(n):
				if n in [0,1]:
					return str(n)
				else:
					return f"\\frac{{ {n.p} }}{{ {n.q} }}"
				

			s = ""
			elems = list(d.items())
			if not elems:
				return s
			initial, rest = elems[0], elems[1:]
			s += f"$\\ket{{ {format_num(initial[1])} }}{{ \\obs{{ obs{initial[0]} }} }}"

			for o, p in rest:
				s += f" + \\ket{{ {format_num(p)} }}{{ \\obs{{ obs{o} }} }}"
			s += "$"

			return s


		nl = "\\\\[5pt]"
		f = "l|" + "l" * len(self.suffixes)
		print(f"\\begin{{tabular}}{{ {f} }}")
		for s in self.suffixes:
			print("&"+suffix_to_string(s))
		print(nl)
		print("\\toprule")

		for p in self.short_prefixes:
			print(prefix_to_string(p), end="")
			for s in self.suffixes:
				trace = p + s[0]
				d = self.get_distribution(trace, s[1])
				print("&" + distribution_to_string(d), end="")
			print(nl)
		print("\\midrule")

		for p in self.long_prefixes:
			print(prefix_to_string(p), end="")
			for s in self.suffixes:
				trace = p + s[0]
				d = self.get_distribution(trace, s[1])
				print("&" + distribution_to_string(d), end="")
			print(nl)
		print("\\bottomrule")
		print("\\end{tabular}")
	
	def get_matrix_row(self, prefix):
		size = (len(self.suffixes)+1)*len(self.mdp.observations)
		#row = sympy.zeros(size)
		vector = np.full(size, Rational(0, 1))
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
		return np.array_equal(self.get_matrix_row(p1), self.get_matrix_row(p2))
	
	def learn_mdp(self):
		iteration = 0
		while True:
			#self.fix_table()
			#self.print_observation_table()
			while self.make_closed():# or self.make_consistent():
				#print("done one iteration of make closed")
				#self.fix_table()
				print(len(self.short_prefixes))
				#print("making closed...")

			h = self.create_hypothesis()
			assert h.check()
			self.check_hypothesis(h)

			self.stats['equivalence_queries'] += 1
			print("looking for cex")
			cex = self.mdp.try_find_counter_example(h, self.config['tries'], self.config['max_observation_length'])
			if cex is None:
				return h
			print("got cex")

			self.process_counter_example(cex)

			iteration += 1
	
	def process_counter_example(self, cex):
		option = self.config['cex']
		if option == 'all_prefixes':
			(initial_output, current_prefix), final_input = cex
			while current_prefix:
				p = initial_output, tuple(current_prefix)
				self.short_prefixes.add(p)
				self.long_prefixes.discard(p)
				current_prefix = current_prefix[:-1]
		elif option == 'shortest_prefix':
			(initial_output, rest), final_input = cex
			minimal_prefix = self.get_minimal_prefix(cex)
			self.short_prefixes.add(minimal_prefix)
			self.long_prefixes.remove(minimal_prefix)
		elif option == 'all_suffixes':
			(initial_output, current_suffix), final_input = cex
			while current_suffix:
				p = tuple(current_suffix), final_input
				self.suffixes.add(p)
				current_suffix = current_suffix[1:]
		elif option == 'longest_suffix':
			(initial_output, current_suffix), final_input = cex
			p = tuple(current_suffix), final_input
			self.suffixes.add(p)
		elif option == 'shortest_suffix':
			minimal_suffix = self.get_minimal_suffix(cex)
			assert minimal_suffix not in self.suffixes
			self.suffixes.add(minimal_suffix)
			for p in self.short_prefixes | self.long_prefixes:
				new_trace = p[0], p[1] + minimal_suffix[0]
				self.get_distribution(new_trace, minimal_suffix[1])

		else: assert False

	
	def get_minimal_prefix(self, cex):
		(initial, current_rest), current_final = cex
		while current_rest:
			next_rest, next_final = current_rest[:-1], current_rest[-1][0]
			prefix = initial, tuple(next_rest)

			if prefix in self.short_prefixes:
				#print("minimal", (initial, tuple(current_rest)))
				return initial, tuple(current_rest)

			current_rest, current_final = next_rest, next_final
		assert False
		return initial, ()

	def get_minimal_suffix(self, cex):
		(initial, p), f = cex
		current_prefix = ()
		current_suffix = tuple(p)

		i = 0
		while (initial, current_prefix) in (self.short_prefixes | self.long_prefixes):
			current_prefix = current_prefix + (p[i],)
			print((initial, current_prefix))
			#print(self.short_prefixes | self.long_prefixes)
			current_suffix = current_suffix[1:]
			assert ((initial, current_prefix + current_suffix), f) == cex

			i += 1
		return tuple(current_suffix), f

	def get_last(self, prefix):
		if prefix[1]:
			return prefix[1][-1][1]
		else:
			return prefix[0]

	@util.start_finish_print
	def create_hypothesis(self):
		if self.config['linear']:
			result = self.create_hypothesis_linear()
		else:
			result = self.create_hypothesis_non_linear()

		return result
	
	def create_hypothesis_non_linear(self):
		states = {}
		prefix_to_row = {}

		sorted_prefixes = sorted(self.short_prefixes, key=lambda k: len(k[1]))
		for p in sorted_prefixes:
			row = self.get_matrix_row(p)
			if util.HashableArray(row) not in states:
				output = self.get_last(p)
				states[util.HashableArray(row)] = mdp.State(f's{len(states)}', output)
			prefix_to_row[p] = row

		for p in self.long_prefixes:
			row = self.get_matrix_row(p)
			assert p not in prefix_to_row
			prefix_to_row[p] = row

		initial_prefix = self.mdp.initial_state.observation, ()
		initial_state = states[util.HashableArray(prefix_to_row[initial_prefix])]

		rows_done = set()
		for p, r in prefix_to_row.items():
			rp = util.HashableArray(r)
			if rp in rows_done or rp not in states:
				continue
			rows_done.add(rp)

			for i, o in itertools.product(self.mdp.inputs, self.mdp.observations):
				extended_trace = p[0], p[1] + ((i, o),)
				prob = self.get_probability(p, i, o)
				if prob == 0 or extended_trace not in prefix_to_row:
					continue
				next_row = prefix_to_row[extended_trace]

				next_state = states[util.HashableArray(next_row)]
				states[rp].add_transition(i, next_state, prob)

		return mdp.MDP(list(states.values()), initial_state, self.mdp.inputs, self.mdp.observations)

	def create_hypothesis_linear(self):
		states = {}
		prefix_to_row = {}
		access_sequence = {}

		for p in self.long_prefixes:
			r = self.get_matrix_row(p)
			prefix_to_row[p] = r

		decomps = self.get_vector_decompositions([self.get_matrix_row(p) for p in self.short_prefixes])
		independent = 0
		for d in decomps:
			if d is None:
				independent += 1
		print("idp", independent)


		prime_rows = []
		prime_rows_by_observation = {o:[] for o in self.mdp.observations}
		for p in self.short_prefixes:
			r = self.get_matrix_row(p)
			output = self.get_last(p)

			access_sequence[util.HashableArray(r)] = p
			prime_rows.append(r)
			prime_rows_by_observation[output].append(r)
			states[util.HashableArray(r)] = mdp.State(f's{len(states)}', output)
			prefix_to_row[p] = r

		initial_prefix = self.mdp.initial_state.observation, ()
		initial_row = prefix_to_row[initial_prefix]
		initial_row_ap = access_sequence[util.HashableArray(initial_row)]
		initial_state = states[util.HashableArray(prefix_to_row[initial_row_ap])]

		for r in prime_rows:
			p = access_sequence[util.HashableArray(r)]
			state = states[util.HashableArray(r)]

			for i, o in itertools.product(self.mdp.inputs, self.mdp.observations):
				extended_trace = p[0], p[1] + ((i, o),)
				p1 = self.get_probability(p, i, o)
				if p1 == 0:
					continue

				next_row = prefix_to_row[extended_trace]
				if extended_trace in self.short_prefixes:
					state.add_transition(i, states[util.HashableArray(next_row)], p1)
				else:
					vector = self.get_matrix_row(extended_trace)
					solution = self.get_probabilistic_decomposition(prime_rows, vector)
					assert solution is not None

					for row, prob in solution.items():
						if prob == 0:
							continue
						#print(state, i, states[row], prob)
						state.add_transition(i, states[row], p1*prob)


		return mdp.MDP(list(states.values()), initial_state, self.mdp.inputs, self.mdp.observations)
	def create_hypothesis_linear2(self):
		states = {}
		prefix_to_row = {}
		access_sequence = {}

		unique_rows = []
		for p in self.short_prefixes:
			r = util.HashableArray(self.get_matrix_row(p))
			if not any(map(lambda v: np.array_equal(v, r.arr), unique_rows)):
				unique_rows.append(r.arr)
				access_sequence[r] = p
				output = self.get_last(p)

			prefix_to_row[p] = r.arr

		for p in self.long_prefixes:
			r = self.get_matrix_row(p)
			prefix_to_row[p] = r

		vector_decomposition = self.get_vector_decompositions(unique_rows)
		prime_rows_by_observation = {o:[] for o in self.mdp.observations}
		prime_rows = []
		for r, d in zip(unique_rows, vector_decomposition):
			if d is None:
				output = self.get_last(p)
				prime_rows_by_observation[output].append(r)
				prime_rows.append(r)
				p = access_sequence[util.HashableArray(r)]
				states[util.HashableArray(r)] = mdp.State(f's{len(states)}', output)

		initial_prefix = self.mdp.initial_state.observation, ()
		initial_row = prefix_to_row[initial_prefix]
		initial_row_ap = access_sequence[util.HashableArray(initial_row)]
		initial_state = states[util.HashableArray(prefix_to_row[initial_row_ap])]

		#for p, r in prefix_to_row.items():
		#print("PR:", prime_rows)
		for r in prime_rows:
			p = access_sequence[util.HashableArray(r)]
			state = states[util.HashableArray(r)]

			for i, o in itertools.product(self.mdp.inputs, self.mdp.observations):
				extended_trace = p[0], p[1] + ((i, o),)
				p1 = self.get_probability(p, i, o)
				if p1 == 0:
					continue
				next_row = prefix_to_row[extended_trace]
				if extended_trace in self.short_prefixes:
					state.add_transition(i, states[util.HashableArray(next_row)], p1)
				else:
					vector = self.get_matrix_row(extended_trace)
					solution = self.get_probabilistic_decomposition(prime_rows_by_observation[o], vector)
					assert solution is not None

					for row, prob in solution.items():
						#print(state, i, states[row], prob)
						state.add_transition(i, states[row], p1*prob)


		return mdp.MDP(list(states.values()), initial_state, self.mdp.inputs, self.mdp.observations)
	
	# Check if the hypothesis matches with the observation table data
	def check_hypothesis(self, h):
		for initial, p1 in self.short_prefixes | self.long_prefixes:
			for p2, final in self.suffixes:
				combined = (initial, p1+p2)
				cell = self.distributions[(combined, final)]

				distr = h.get_exact_output_distribution_io(combined, final)
				#print(distr, "vs", cell)
				assert util.distribution_eq(distr, cell)

	
	#@util.start_finish_print
	#def get_probabilistic_decomposition(self, vectors, vector):
	#	# Get unique vectors and establish an ordering
	#	unique_vectors = list(set([util.HashableArray(v) for v in vectors]))
	#	vector = np.hstack([vector, [1]])
	#	matrix = np.array([v.arr for v in unique_vectors]).T
	#	vector_count = matrix.shape[1]
	#	matrix = np.vstack([matrix, np.ones(vector_count)])
	#	solution = optimize.linprog(np.zeros(vector_count), A_eq=matrix, b_eq=vector, method='simplex')

	#	if solution.success:
	#		probs = { unique_vectors[i]: solution.x[i] for i in range(len(unique_vectors)) }
	#		return probs
	#	else:
	#		return None

	def get_probabilistic_decomposition(self, vectors, vector):
		# Get unique vectors and establish an ordering
		#unique_vectors = list(set([util.HashableArray(v) for v in vectors]))
		#vector = np.hstack([vector, [1]])
		matrix = np.array(vectors).T
		vector_count = matrix.shape[1]
		#matrix = np.vstack([matrix, np.ones(vector_count)])
		solution = optimize.linprog(np.zeros(vector_count), A_eq=matrix, b_eq=vector, method='simplex')

		if solution.success:
			probs = { util.HashableArray(vectors[i]): solution.x[i] for i in range(len(vectors)) }
			return probs
		else:
			return None

	#def get_probabilistic_decomposition3(self, vectors, vector):
	#	# Get unique vectors and establish an ordering
	#	matrix = sympy.Matrix()
	#	for v in vectors:
	#		matrix.col_insert(matrix.shape[1], v)

	#	vector_count = matrix.shape[1]
	#	solution = util.linprog(matrix, vector)

	#	if solution is not None:
	#		probs = { vectors[i]: solution.x[i] for i in range(len(unique_vectors)) }
	#		return probs
	#	else:
	#		return None
	
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
