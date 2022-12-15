include("valfun.jl")
include("valfun_lib.jl")
include("graph_utils.jl")

# use_complement = true
# graph_name = "san200-0-7-2"
# G = load_dimacs_graph(graph_name, use_complement)

using DelimitedFiles
use_complement = false
graph_name = "ivan-6-bad"
family = "chordal"
G = load_family_graph(graph_name, family, use_complement)

# savegraph(graph_name * ".lgz", G)

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
# w = ones(n)
w = [2, 1, 1, 1, 1, 1]
# w = rand(n)

# sol = max_clique(G, w)
# println(z)
# z_set = findall(z .> 0.1)
# println(is_clique(G, z_set))
# plot_graph(G, graph_name, use_complement; S_to_color=z_set, suffix="test_qstab", add_label=true)

sol = clique_stable_set_lp(G, w)
θ = sol.value
val = valfun_qstab(sol.λ, sol.cliques)
print_valfun(val, n)
val_qstab_sdp = valfun(qstab_to_sdp(G, w, sol.λ, sol.cliques))
print_valfun(val_qstab_sdp, n)

# sol = dualSDP(G, w; solver="Mosek", ϵ=1e-12)
# θ = sol.value
# println("SDP Value: ", θ)
# Q = Matrix(sol.Q)
# display(Q)
# val = valfun(Q)

# x_stable, _ = tabu_valfun(G, w, θ, val; ϵ=1e-7)
# println(findall(x_stable))
# println("Retrieved value: ", w' * x_stable)
#
tabu_valfun_test(G, w, θ, val_qstab_sdp; ϵ=1e-6, solver="Mosek", solver_ϵ=1e-9, verbose=false)

# println("Verifying subadditivity...")
# test_subadditivity(θ, 1:n, val; ϵ=1e-6)
# random_test_subadditivity(Q, θ; solver="COPT", n_iter=100000, ϵ=1e-6)
#
# println("Verifying subadditivity after discarding...")
# S = collect(1:n)
# vertex_value_discard!(w, val, S; ϵ=1e-7)
# fixed_point_discard!(G, w, θ, val, S; ϵ=1e-7)
# test_subadditivity(Q, θ, S; solver="Mosek", ϵ=1e-5, solver_ϵ=1e-12)
# random_test_subadditivity(Q, θ, S; solver="COPT", n_iter=100000, ϵ=1e-4)
