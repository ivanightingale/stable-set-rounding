include("valfun.jl")
include("valfun_lib.jl")
include("graph_utils.jl")

# use_complement = true
# graph_name = "san200-0-7-2"
# G = load_dimacs_graph(graph_name, use_complement)

use_complement = false
graph_name = "ivan-6-bad"
family = "chordal"
G = load_family_graph(graph_name, family, use_complement)

# plot_graph(G, graph_name, use_complement)

n = nv(G)
println(graph_name, " ", use_complement)
println(n)
println(ne(G))

# Weight Vector
# w = ones(n)
w = [2, 1, 1, 1, 1, 1]  # for ivan-6-bad
# w = [1, 1, 1, 1, 2, 2, 1]  # for ivan-7-bad

sol = qstab_lp(G, w; verbose=false)
θ = sol.value
println(θ)
failure_count = test_qstab_valfuns(G, w, θ, sol.λ_ext_points, sol.cliques; use_theta=false)
println("Number of failed extreme points: ", failure_count, "; total extreme points: ", length(sol.λ_ext_points))

λ_interior = sum(sol.λ_ext_points) / length(sol.λ_ext_points)
val = valfun_qstab(λ_interior, sol.cliques)
val_qstab_sdp = valfun(qstab_to_sdp(G, w, λ_interior, sol.cliques))
tabu_valfun_test(G, w, θ, val; use_theta=false, ϵ=1e-6, solver="Mosek", solver_ϵ=1e-9, verbose=false)
tabu_valfun_test(G, w, θ, val_qstab_sdp; use_theta=false, ϵ=1e-6, solver="Mosek", solver_ϵ=1e-9, verbose=false)


# sol = dualSDP(G, w; solver="Mosek", ϵ=1e-12)
# θ = sol.value
# println("SDP Value: ", θ)
# Q = Matrix(sol.Q)
# display(Q)
# val = valfun(Q)

# x_stable, _ = tabu_valfun(G, w, θ, val; ϵ=1e-7)
# println(findall(x_stable))
# println("Retrieved value: ", w' * x_stable)
