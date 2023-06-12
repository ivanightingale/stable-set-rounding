include("src/ValFun/valfun.jl")
using .ValFun
include("src/valfun_algorithms.jl")
include("src/valfun_experiments.jl")
include("src/graph_utils.jl")

sdp_params = Dict(
    :formulation => :grotschel,
    :solve_dual => true,  # whether to formulate the Model as the primal or the dual problem.
                          # Experiments show that COPT is much faster in solving the primal if G
                          # is dense, and in solving the dual if G is sparse.
    :solver => :COPT,
    :solver_ϵ => 0,  # solver gap tolerance, 0 for solver default
    :solver_feas_ϵ => 0,  # solver feasibility tolerance, 0 for solver default
    :valfun_ϵ => 1e-6,  # valfun tolerance in discarding rules, etc.
    :use_div => true,  # whether to use \ in computing valfun
    :pinv_rtol => 1e-9,  # rtol value in pinv()
    :verbose => true,
)

qstab_params = Dict(
    :solver => :COPT,
    :solver_ϵ => 0,
    :solver_feas_ϵ => 0,
    :valfun_ϵ => 1e-6,
    :use_all_cliques => true,
    :verbose => true,
)

use_complement = true
graph_name = "p-hat1000-3"
G = load_dimacs_graph(graph_name, use_complement)

# use_complement = false
# graph_name = "diego-11"
# family = "perfect"
# G = load_family_graph(graph_name, family, use_complement)

# use_complement = false
# G, graph_name = generate_family_graph("hole", 10, use_complement; k=3)

# use_complement = false
# graph_name = "gsg_2000_1"
# G = load_gsg_graph(graph_name, use_complement)

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


run_round_valfun(G, w, sdp_params)
# run_tabu_valfun_test(G, w, sdp_params)
# run_tabu_valfun_compare(G, w, sdp_params, qstab_params)
# run_test_qstab_valfuns(G, w, qstab_params, sdp_params)
# run_find_bad_valfun(G, sdp_params)
