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
	assert m.check()
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
	assert m.check()
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
	assert m.check()
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
	m = MDP([s0,s1,s2,s3,s4], s0, actions, [0,1,2])
	assert m.check()
	return m


if __name__ == '__main__':
	print("Wrong file!")
