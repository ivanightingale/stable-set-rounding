using MatrixMarket
using Graphs

graph_file = "san200-0-7-1.mtx";

function load_graph()
	A = mmread("../dat/" * graph_file)
	G = complement(SimpleGraph(A))
	n = nv(G)
	i0 = n + 1
	return G, n, i0
end
