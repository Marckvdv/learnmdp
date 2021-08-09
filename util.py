import numpy as np
import math

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
