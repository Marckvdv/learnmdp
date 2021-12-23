import numpy as np
import math
import sympy

sympy.init_printing(use_unicode=False)

# Cheaty hashable numpy array
class HashableArray:
	def __init__(self, arr):
		self.arr = arr
	
	def __hash__(self):
		return hash(tuple(self.arr))
	
	def __eq__(self, other):
		return (self.arr == other.arr).all()
	
	def __repr__(self):
		return repr(self.arr)
	
	def __str__(self):
		return str(self.arr)

def distribution_eq(a, b, epsilon=1e-3):
	for k, v in a.items():
		if v > 0 and (k not in b or not math.isclose(b[k], v, rel_tol=epsilon)):
			return False
	return True
	
current_depth = 0
def start_finish_print(func):
	def wrapper(*args, **kwargs):
		#global current_depth
		#print("{}>>> {}".format('\t'*current_depth, func.__name__))
		#current_depth += 1
		result = func(*args, **kwargs)
		#current_depth -= 1
		#print("{}<<< {}".format('\t'*current_depth, func.__name__))

		return result
	return wrapper

def linprog(A_eq, b_eq):
	def cmp_le(a, b):
		return b is None or isinstance(b, sympy.core.numbers.ComplexInfinity) or a < b

	# First put the matrix in standard form
	tableau = sympy.Matrix(A_eq)
	i = sympy.eye(tableau.shape[0])
	for r in range(i.shape[0]):
		tableau = tableau.col_insert(tableau.shape[1], i.col(r))
	tableau = tableau.col_insert(tableau.shape[1], b_eq)
	c = sympy.zeros(1, tableau.shape[1])

	for i in range(A_eq.shape[1], A_eq.shape[1] + A_eq.shape[0]):
		c[i] = sympy.Rational(-1)

	tableau = tableau.row_insert(tableau.shape[0], c)

	last_row_idx = tableau.shape[0]-1
	rows, columns = tableau.shape

	for i in range(last_row_idx):
		tableau[last_row_idx,:] += tableau[i,:]
	
	#print("initial tableau")
	#print(sympy.pretty(tableau))
	while True:
		# Find pivot column
		pivot_col = None
		for i in range(columns-1):
			if tableau[last_row_idx, i] > 0:
				pivot_col = i
				break

		if pivot_col is None:
			break

		# Find pivot row
		pivot_row = None
		pivot_row_min = None
		for j in range(rows-1):
			if tableau[j, pivot_col] == 0:
				if pivot_row is None:
					pivot_row = j
					pivot_row_min = float('inf')
			else:
				d = tableau[rows-1, pivot_col] / tableau[j, pivot_col]			
				if cmp_le(d, pivot_row_min):
					pivot_row = j
					pivot_row_min = d

		if pivot_row is None:
			break

		#print("pivot element", pivot_row, pivot_col)
		tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]	
		for i in range(rows):
			if i == pivot_row:
				continue

			tableau[i, :] -= tableau[pivot_row, :] * tableau[i, pivot_col]
	
	x = sympy.zeros(1, A_eq.shape[1])
	
	for i in range(rows-1):
		found = False
		for j in range(A_eq.shape[1]):
			if tableau[i, j] == 1:
				x[j] = tableau[i, columns-1]
				found = True
				break
		if not found:
			return None
	return x

#M = sympy.Matrix
#R = sympy.Rational
#
#m1 = M([[0,1,0,R(1,3)],[0,0,1,R(2,3)]])
#print(sympy.pretty(m1))
#print()
#
#m = M([[R(1,2), 1, 0], [R(1,2), 0, 1]])
#v = M([R(1,3), R(2,3)])
#v1 = M([R(1,3), R(1,3), R(1,3)])
#sol = linprog(m1, v).T
#print(sympy.pretty(sol))
#print()
#print(sympy.pretty(m1 @ sol))
