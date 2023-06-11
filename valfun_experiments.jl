include("src/ValFun/valfun.jl")
using .ValFun
include("src/valfun_algorithms.jl")
include("src/graph_utils.jl")

# use_complement = true
# graph_name = "hamming8-2"
# G = load_dimacs_graph(graph_name, use_complement)

use_complement = false
graph_name = "diego-11"
family = "perfect"
G = load_family_graph(graph_name, family, use_complement)

# use_complement = false
# G, graph_name = generate_family_graph("hole", 10, use_complement; k=3)

# plot_graph(G, graph_name, use_complement; add_label=true)
# display(Matrix(adjacency_matrix(G)))

n = nv(G)
println(graph_name, " ", use_complement)
println(n)
println(ne(G))

# Weight Vector
# w = ones(n)
# w = rand(1:10, n)
# println(w)
# w = [2, 1, 1, 1, 1, 1]  # for ivan-6
# w = [1, 1, 1, 1, 2, 2, 8, 1, 1, 1, 1, 2, 1, 5, 7]  # for connecting-15-1.co
w = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # for diego-11
# w = [1, 1, 1, 1, 2, 2, 1]  # for ivan-7
# w = [6, 1, 1, 1, 3.5, 3.5, 1, 3.5, 1, 4, 1, 1, 4, 2.5, 1]  # for connecting-15-2.co
