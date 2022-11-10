using MatrixMarket
using Graphs

graph_file = "hamming6-2.mtx";

function load_graph()
	A = mmread("../dat/" * graph_file)
	G = complement(SimpleGraph(A))
	n = nv(G)
	i0 = n + 1
	return G, n, i0
end
