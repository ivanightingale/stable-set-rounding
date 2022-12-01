include("valfun.jl")
include("graph_utils.jl")

use_complement = true
graph_name = "san400-0-7-3"
G = load_dimacs_graph(graph_name, use_complement)

# using DelimitedFiles
# use_complement = false
# graph_name = "ivan-7-subadditive"
# family = "random"
# G = load_family_graph(graph_name, family, use_complement)

# plot_graph(G, graph_name, use_complement)

# use_complement = false
# G, graph_name = generate_family_graph("hole", 5, use_complement)

println(graph_name, " ", use_complement)

# plot_graph(G, graph_name, use_complement)

# display(adjacency_matrix(G))

n = nv(G)
println(n)
println(ne(G))

# Weight Vector
w = ones(n)
# w = 10 * rand(n)

sol = dualSDP(collect(edges(G)), w; solver="COSMO", ϵ=1e-9)
θ = sol.value
println("SDP Value: ", θ)
Q = Matrix(sol.Q)
val = valfun(Q)
# display(Q)
# val = valfun_sdp(Q; solver="Mosek")
# print_valfun(val, n, 4)

# x_stable, _ = tabu_valfun(G, w, θ, val; ϵ=1e-7)
# println(findall(x_stable))
# println("Retrieved value: ", w' * x_stable)

tabu_valfun_test(G, w, θ, val; ϵ=1e-6, verbose=false)

# println("Verifying subadditivity...")

# test_subadditivity(Q, θ; solver="Mosek", ϵ=1e-6)
# random_test_subadditivity(Q, θ; solver="Mosek", n_iter=100000)

# S = collect(1:n)
# vertex_value_discard!(w, val, S; ϵ=1e-7, verbose=true)
# fixed_point_discard!(G, w, θ, val, S; ϵ=1e-7, verbose=true)
# test_subadditivity(Q, θ, S; solver="Mosek", ϵ=1e-5)
# random_test_subadditivity(Q, θ, S; solver="Mosek", n_iter=100000)
