include("valfun.jl")

# using MatrixMarket
# use_complement = true
# graph_name = "hamming8-2"
# G = load_dimacs_graph(graph_name, use_complement)

using DelimitedFiles
use_complement = true
graph_name = "pruned-15-5"
family = "chordal"
G = load_family_graph(graph_name, family, use_complement)

# use_complement = false
# G, graph_name = generate_family_graph("wheel", 7, use_complement)

n = nv(G)
println(n)
println(ne(G))

# Weight Vector
w = ones(n)

val, θ = get_valfun(G, w)

# println("round_valfun begins")
# xr = round_valfun(G, w, val, θ)
# println(findall(xr))
# println("Rounded Value: ", w' * xr)


# println("tabu_valfun_imperfect begins")
# xt = tabu_valfun_imperfect(G, w, val, θ, 1e-3)
# println(findall(xt))
# println("Rounded Value: ", w' * xt)

perfect_tabu_valfun_verify_I(G, w, val, θ, 1e-3, graph_name, use_complement)
