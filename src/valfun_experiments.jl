include("valfun.jl")
include("graph_utils.jl")

use_complement = true
graph_name = "san400-0-7-3"
G = load_dimacs_graph(graph_name, use_complement)

# using DelimitedFiles
# use_complement = true
# graph_name = "connecting-15-1"
# family = "chordal"
# G = load_family_graph(graph_name, family, use_complement)

# plot_graph(G, graph_name, use_complement)

# use_complement = false
# G, graph_name = generate_family_graph("hole", 5, use_complement)

println(graph_name, " ", use_complement)

# plot_graph(G, graph_name, use_complement)

n = nv(G)
println(n)
println(ne(G))

# Weight Vector
w = ones(n)
# w = 10 * rand(n)

sol = dualSDP(collect(edges(G)), w; solver="COSMO", ϵ=1e-8)
θ = sol.value
println("SDP Value: ", θ)

Q = Matrix(sol.Q)
val = valfun(Q)
# val = valfun_sdp(Q; solver="Mosek")
# print_valfun(val, n, 4)

# xr = round_valfun(G, w, val, θ)

x_stable, _ = tabu_valfun(G, w, θ, val; ϵ=1e-4, verbose=true)
# #
println(findall(x_stable))
println("Rounded Value: ", w' * x_stable)

# tabu_valfun_test(G, w, θ, val; ϵ=1e-6)

# println("Verifying subadditivity...")
# test_subadditivity(sol; solver="Mosek", ϵ=1e-4)
# random_test_subadditivity(sol; solver="Mosek", n_iter=10000)
