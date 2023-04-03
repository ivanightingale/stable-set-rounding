include("valfun_utils.jl")
include("sdp_valfun.jl")
include("qstab_valfun.jl")
include("valfun_algorithms.jl")
include("graph_utils.jl")

use_complement = true
graph_name = "hamming10-2"
G = load_dimacs_graph(graph_name, use_complement)

# use_complement = false
# graph_name = "ivan-6"
# family = "perfect"
# G = load_family_graph(graph_name, family, use_complement)
# G, graph_name = generate_family_graph("path", 5, use_complement; k=3)

# plot_graph(G, graph_name, use_complement; add_label=true)
# display(Matrix(adjacency_matrix(G)))

n = nv(G)
println(graph_name, " ", use_complement)
println(n)
println(ne(G))

# Weight Vector
w = ones(n)
# w = rand(1:10, n)
# println(w)
# w = [2, 1, 1, 1, 1, 1]  # for ivan-6
# w = [1, 1, 1, 1, 2, 2, 8, 1, 1, 1, 1, 2, 1, 5, 7]  # for connecting-15-1.co
# w = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # for diego-11
# w = [1, 1, 1, 1, 2, 2, 1]  # for ivan-7
# w = [6, 1, 1, 1, 3.5, 3.5, 1, 3.5, 1, 4, 1, 1, 4, 2.5, 1]  # for connecting-15-2.co

######################
# SDP formulations performance comparison
######################
# dualSDP(G, w, true; solver="Mosek")  # warm-up
# for solver in ["Mosek", "SCS", "COSMO", "COPT"]
# 	for solve_dual in [true, false]
# 		println(solver)
# 		println(solve_dual)
# 		@time sdp_sol = dualSDP(G, w, solve_dual; solver=solver, ϵ=1e-7)
# 		θ = sdp_sol.value
# 		println("SDP Value: ", θ)
# 		# display(sdp_sol.Q)
# 	end
# end

formulation = "lovasz"
println(formulation)
@time sdp_sol = dualSDP(G, w, true, formulation; solver="COSMO", ϵ=0, verbose=false)
θ = sdp_sol.value
println("SDP Value: ", θ)
# Q = Matrix(sdp_sol.Q)
# val = valfun(Q)
# x_stable, _ = tabu_valfun(G, w, θ, val)
# println("Retrieved value: ", w' * x_stable)

######################
# QSTAB LP interior point value function vs. SDP value function
######################
# sdp_sol = dualSDP(G, w; solver="Mosek", ϵ=1e-12)
# θ = sdp_sol.value
# println("SDP Value: ", θ)
# Q = Matrix(sdp_sol.Q)
# val_sdp = valfun(Q)
# # tabu_valfun_test(G, w, sdp_sol.value, val_sdp; use_theta=false, ϵ=1e-8, verbose=true)

# qstab_sol = qstab_lp_interior_point(G, w; use_all_cliques=true, solver="Mosek", ϵ=1e-12, verbose=false)
# println("QSTAB LP value: ", qstab_sol.value)
# λ_interior = qstab_sol.λ
# val_qstab = valfun_qstab(λ_interior, qstab_sol.cliques)
# # tabu_valfun_test(G, w, qstab_sol.value, val_qstab; use_theta=false, ϵ=1e-6, verbose=true)

# tabu_valfun_compare(G, w, θ, val_sdp, val_qstab; ϵ=1e-8, verbose=true)

######################
# QSTAB LP extreme points value functions verification
######################
# qstab_sol = qstab_lp(G, w; use_all_cliques=false, ϵ=1e-9, verbose=false)
# θ = qstab_sol.value
# println("QSTAB LP value: ", θ)
# failure_count = test_qstab_valfuns(G, w, θ, qstab_sol.λ_ext_points, qstab_sol.cliques; use_theta=false, verbose=false)
# println("Number of failed extreme points: ", failure_count, "; total extreme points: ", length(qstab_sol.λ_ext_points))
# λ_interior = sum(qstab_sol.λ_ext_points) / length(qstab_sol.λ_ext_points)
# val = valfun_qstab(λ_interior, qstab_sol.cliques)
# val_qstab_sdp = valfun(qstab_to_sdp(G, w, λ_interior, qstab_sol.cliques))
# println("Testing the interior point...")
# println(λ_interior)
# tabu_valfun_test(G, w, θ, val; use_theta=false, ϵ=1e-6, solver="Mosek", solver_ϵ=0, verbose=true)
# println(tabu_valfun_test(G, w, θ, val_qstab_sdp; use_theta=false, ϵ=1e-6, solver="Mosek", solver_ϵ=1e-9, verbose=false))


######################
# SDP to QSTAB conversion
######################
# sdp_sol = dualSDP(G, w; solver="Mosek", ϵ=1e-12)
# θ = sdp_sol.value
# println("SDP Value: ", θ)
# Q = Matrix(sdp_sol.Q)
# display(Q)
# println()
# val = valfun(Q)
# tabu_valfun_test(G, w, θ, val; use_theta=false, verbose=true)
# println(sdp_to_qstab(Q, w, all_cliques(G); solver="COPT"))



######################
# tabu_valfun
######################
# x_stable, _ = tabu_valfun(G, w, θ, val; ϵ=1e-7)
# println(findall(x_stable))
# println("Retrieved value: ", w' * x_stable)
