from mdp import *

import sympy

def get_float_n(n):
	states = [State(f's{i}', 0) for i in range(n)]
	b = State('b', 1)
	e = State('e', 0)
	h = sympy.Rational(1,2)

	states[0].add_transition('f', states[0], h)
	states[0].add_transition('r', b, sympy.Rational(1,1))
	for i in range(n-1):
		states[i].add_transition('f', states[i+1], h)
		states[i+1].add_transition('f', states[i], h)
		states[i+1].add_transition('r', e, sympy.Rational(1,1))
	states[n-1].add_transition('f', states[n-1], h)
	for s in [b, e]:
		for i in ['f', 'r']:
			s.add_transition(i, states[0], sympy.Rational(1,1))

	m = MDP(states + [b] + [e], states[0], ['f', 'r'], [0,1])
	return m

def get_simple():
	s1 = State('s1', 1)
	s2 = State('s2', 2)
	s3 = State('s3', 3)

	h = sympy.Rational(1,2)
	s1.add_transition('f', s2, h)
	s1.add_transition('f', s3, h)

	s2.add_transition('f', s1, h)
	s2.add_transition('f', s3, h)

	s3.add_transition('f', s1, h)
	s3.add_transition('f', s2, h)

	m = MDP([s1,s2,s3], s1, ['f'], [1,2,3])
	return m

def get_simple2():
	s0 = State('s0', 0)
	s1 = State('s1', 0)
	s2 = State('s2', 0)
	s3 = State('s3', 1)
	s4 = State('s4', 2)

	one = sympy.Rational(1,1)
	t = sympy.Rational(1,4)
	s0.add_transition('l', s1, one)
	s0.add_transition('r', s2, one)
	s0.add_transition('g', s1, t)
	s0.add_transition('g', s2, 1-t)

	for state1, state2 in [(s1, s3), (s2, s4), (s3, s0), (s4, s0)]:
		for a in ['l','r','g']:
		#for a in ['g']:
			state1.add_transition(a, state2, one)
	m = MDP([s0,s1,s2,s3,s4], s0, ['l','r','g'], [0,1,2])
	return m
	
def get_simple2n(n=1):
	s0 = State('s0', 0)
	s1 = State('s1', 0)
	s2 = State('s2', 0)
	s3 = State('s3', 1)
	s4 = State('s4', 2)

	actions = [chr(ord('a')+i) for i in range(2+n)]
	one = sympy.Rational(1,1)
	s0.add_transition(actions[0], s1, one)
	s0.add_transition(actions[1], s2, one)

	for i in range(n):
		action = actions[2+i]
		prob = sympy.Rational(1,i+2)
		
		s0.add_transition(action, s1, prob)
		s0.add_transition(action, s2, 1-prob)

	for state1, state2 in [(s1, s3), (s2, s4), (s3, s0), (s4, s0)]:
		for a in actions:
		#for a in ['g']:
			state1.add_transition(a, state2, one)
	m = MDP([s0,s1,s2,s3,s4], s0, actions, [0,1,2], f'simple_{n}')
	return m

def get_chain(n=3):
	# assumed to be of sufficient length
	odd_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 71]
	start_state = State('s0', 0)
	left_states, right_states, end_states = [], [], []
	one = sympy.Rational(1,1)
	half = sympy.Rational(1,2)
	for i in range(n):
		left_states.append(State(f's{i}_l', i+1))
		right_states.append(State(f's{i}_r', i+1))
		end_states.append(State(f's{i}_e', i))
	
	actions = ['a', 'b', 'c', 'd']
	start_state.add_transition('a', left_states[0], one)
	start_state.add_transition('b', right_states[0], one)
	start_state.add_transition('c', left_states[0], half)
	start_state.add_transition('c', right_states[0], half)
	start_state.add_transition('d', start_state, one)

	for i in range(n-1):
		p1 = sympy.Rational(1, odd_primes[i])
		p2 = 1-p1

		left_states[i].add_transition('a', left_states[i+1], one)
		left_states[i].add_transition('b', right_states[i+1], one)

		right_states[i].add_transition('a', right_states[i+1], one)
		right_states[i].add_transition('b', left_states[i+1], one)

		left_states[i].add_transition('c', left_states[i+1], p1)
		left_states[i].add_transition('c', right_states[i+1], p2)

		right_states[i].add_transition('c', right_states[i+1], p1)
		right_states[i].add_transition('c', left_states[i+1], p2)

	
	for i in range(n):
		left_states[i].add_transition('d', end_states[i], one)
		right_states[i].add_transition('d', end_states[n-1-i], one)
		for a in actions:
			end_states[i].add_transition(a, end_states[i], one)

	for a in actions[:-1]:
		for s in [left_states[-1], right_states[-1]]:
			s.add_transition(a, start_state, one)

	return MDP([start_state] + left_states + right_states + end_states, start_state, actions, [i for i in range(n+1)], f'chain_{n}')

def get_test1():
	s1 = State('s1', 0)
	s2 = State('s2', 0)
	s3 = State('s3', 1)

	x = sympy.Rational(1,3)
	s1.add_transition('a', s2, x)
	s1.add_transition('a', s1, 1-x)
	s2.add_transition('a', s3, 1)
	s3.add_transition('a', s3, 1)

	return MDP([s1,s2,s3], s1, ['a'], [0,1])

def get_test2():
	s1 = State('s1', 0)
	s2 = State('s2', 0)
	s3 = State('s3', 1)

	x = sympy.Rational(1,3)
	s1.add_transition('a', s2, 1)
	s2.add_transition('a', s2, 1-x)
	s2.add_transition('a', s3, x)
	s3.add_transition('a', s3, 1)

	return MDP([s1,s2,s3], s1, ['a'], [0,1])

def get_random_deterministic(state_count=10, input_count=3, observation_count=5):
	assert observation_count < state_count
	inputs = [ i for i in range(input_count) ]
	observations = [ o for o in range(observation_count) ]
	states = []

	for o in observations:
		s = State(f's{len(states)}', o)
		states.append(s)
	
	for _ in range(len(states), state_count):
		s = State(f's{len(states)}', random.choice(observations))
		states.append(s)

	initial_state = states[0]
	states_by_observation = {o: [] for o in observations}
	for s in states:
		states_by_observation[s.observation].append(s)
	
	for s1 in states:
		for i in inputs:
			transition_count = random.randrange(1, len(observations))
			picked_observations = random.sample(observations, transition_count)
			picked_states = [random.choice(states_by_observation[o]) for o in picked_observations]
			picked_probs = np.random.dirichlet(np.ones(len(picked_states)), size=1)
			for s2, p in zip(picked_states, picked_probs[0]):
				s1.add_transition(i, s2, p)


	return MDP(states, initial_state, inputs, observations)

if __name__ == '__main__':
	print("Wrong file!")
