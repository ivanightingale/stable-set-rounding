
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

# formulation = :grotschel
# println(formulation)
# sdp_out = get_valfun(G, w; solver=:COPT, ϵ=1e-9)
# sdp_sol = sdp_out.sol
# θ = sdp_sol.value
# println("SDP Value: ", θ)
# val = sdp_out.val
# # @time is_success = tabu_valfun_test(G, w, θ, val; use_theta=false, verbose=false)
# # println(is_success)
# @time x_stable, _ = tabu_valfun(G, w, θ, val)
# println("Retrieved value: ", w' * x_stable)

######################
# QSTAB LP interior point value function vs. SDP value function
######################
# sdp_out = get_valfun(G, w; solver=:COPT, ϵ=1e-9)
# sdp_sol = sdp_out.sol
# θ = sdp_sol.value
# println("SDP Value: ", θ)
# val_sdp = sdp_out.val
# # tabu_valfun_test(G, w, sdp_sol.value, val_sdp; use_theta=false, ϵ=1e-8, verbose=true)

# qstab_out = get_valfun_qstab(G, w, true, true)
# qstab_sol = qstab_out.sol
# println("QSTAB LP value: ", qstab_sol.value)
# val_qstab = qstab_out.val
# # tabu_valfun_test(G, w, qstab_sol.value, val_qstab; use_theta=false, ϵ=1e-6, verbose=true)

# tabu_valfun_compare(G, w, θ, val_sdp, val_qstab; ϵ=1e-6, verbose=true)

######################
# QSTAB LP extreme points value functions verification
######################
qstab_sol = qstab_lp_ext(G, w, false; ϵ=1e-9, verbose=false)
θ = qstab_sol.value
println("QSTAB LP value: ", θ)
failure_count = test_qstab_valfuns(G, w, θ, qstab_sol.λ_ext_points, qstab_sol.cliques; use_theta=false, verbose=false)
println("Number of failed extreme points: ", failure_count, "; total extreme points: ", length(qstab_sol.λ_ext_points))
λ_interior = sum(qstab_sol.λ_ext_points) / length(qstab_sol.λ_ext_points)
val = valfun_qstab(λ_interior, qstab_sol.cliques)
val_qstab_sdp = valfun(qstab_to_sdp(G, w, λ_interior, qstab_sol.cliques))
println("Testing the interior point...")
println(λ_interior)
tabu_valfun_test(G, w, θ, val; use_theta=false, ϵ=1e-6, solver="Mosek", solver_ϵ=0, verbose=true)
println(tabu_valfun_test(G, w, θ, val_qstab_sdp; use_theta=false, ϵ=1e-6, solver="Mosek", solver_ϵ=1e-9, verbose=false))
