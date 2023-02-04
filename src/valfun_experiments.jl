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
# G, graph_name = generate_family_graph("hole", 6, use_complement)

# plot_graph(G, graph_name, use_complement)

n = nv(G)
println(graph_name, " ", use_complement)
println(n)
println(ne(G))

# Weight Vector
w = ones(n)
# w = [2, 1, 1, 1, 1, 1]  # for ivan-6-bad
# w = [1, 1, 1, 1, 2, 2, 8, 1, 1, 1, 1, 2, 1, 5, 7]  # for connecting-15-1.co
# w = [1, 1, 1, 1, 2, 2, 1]  # for ivan-7-bad
# w = [6, 1, 1, 1, 3.5, 3.5, 1, 3.5, 1, 4, 1, 1, 4, 2.5, 1]  # for connecting-15-2.co

qstab_sol = qstab_lp(G, w; verbose=false)
θ = qstab_sol.value
println(θ)
failure_count = test_qstab_valfuns(G, w, θ, qstab_sol.λ_ext_points, qstab_sol.cliques; use_theta=true)
println("Number of failed extreme points: ", failure_count, "; total extreme points: ", length(qstab_sol.λ_ext_points))
λ_interior = sum(qstab_sol.λ_ext_points) / length(qstab_sol.λ_ext_points)
val = valfun_qstab(λ_interior, qstab_sol.cliques)
val_qstab_sdp = valfun(qstab_to_sdp(G, w, λ_interior, qstab_sol.cliques))
println("Testing the interior point...")
tabu_valfun_test(G, w, θ, val; use_theta=false, ϵ=1e-6, solver="Mosek", solver_ϵ=1e-9, verbose=true)
tabu_valfun_test(G, w, θ, val_qstab_sdp; use_theta=false, ϵ=1e-6, solver="Mosek", solver_ϵ=1e-9, verbose=false)


# sdp_sol = dualSDP(G, w; solver="Mosek", ϵ=1e-12)
# θ = sdp_sol.value
# println("SDP Value: ", θ)
# Q = Matrix(sdp_sol.Q)
# display(Q)
# val = valfun(Q)
# tabu_valfun_test(G, w, θ, val)

# Q_qstab = qstab_to_sdp(G, w, λ_interior, qstab_sol.cliques)
# println(λ_interior)
# display(Q_qstab)
# sdp_to_qstab(Q_qstab, w, qstab_sol.cliques; solver="Mosek")
# sdp_to_qstab(Q, w, qstab_sol.cliques; solver="Mosek")

# x_stable, _ = tabu_valfun(G, w, θ, val; ϵ=1e-7)
# println(findall(x_stable))
# println("Retrieved value: ", w' * x_stable)
