import matplotlib.pyplot as plt
from pathlib import Path

Path("graphs").mkdir(exist_ok=True)

def graph_double_line(x, y1, y2, xaxis="n", yaxis="", title=""):
	plt.figure()
	plt.plot(x, y1, '-o', label='normal')
	plt.plot(x, y2, '-o', label='linear')

	plt.xticks(x)
	plt.legend()
	plt.grid()
	plt.xlabel(xaxis)
	plt.ylabel(yaxis)
	plt.title(title)

def graph1():
	x = list(range(1, 11))
	y1 = [3, 11, 21, 39, 69, 127, 225, 405, 707, 1309]
	y2 = [3,  7, 10, 13, 16,  19,  22,  25,  28,   31]

	graph_double_line(x, y1, y2, yaxis="states", title="State count versus n (chain)")
	plt.savefig('graphs/graph1.png')

def graph2():
	x = list(range(1, 12))
	y1 = [5, 5, 5, 5,  5,  5,  5,  5,  5,  5,  5]
	y2 = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

	graph_double_line(x, y1, y2, yaxis="states", title="State count versus n (simple)")
	plt.savefig('graphs/graph2.png')

def graph3():
	x = list(range(1, 11))
	y1 = [52, 136, 297, 332, 466, 488, 1135, 584, 787, 1948]
	y2 = [52, 196, 480, 732, 1670, 2468, 5427, 8004, 18695, 26060]

	graph_double_line(x, y1, y2, yaxis="membership queries", title="Membership queries versus n (chain)")
	plt.savefig('graphs/graph3.png')

#def graph4():
#	x = list(range(1, 11))
#	y1 = [48, 84, 180, 
#	y2 = [
#
#	graph_double_line(x, y1, y2, yaxis="membership queries", title="Membership queries versus n (chain)")
#	plt.savefig('graphs/graph4.png')

graph1()
graph2()
graph3()
graph4()
