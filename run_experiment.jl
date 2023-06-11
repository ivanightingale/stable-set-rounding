include("src/ValFun/valfun.jl")
using .ValFun
include("src/valfun_algorithms.jl")
include("src/valfun_experiments.jl")
include("src/graph_utils.jl")

sdp_params = Dict(
    :formulation => :grotschel,
    :solve_dual => true,
    :solver => :COPT,
    :solver_ϵ => 0,  # solver gap tolerance
    :solver_feas_ϵ => 0,  # solver feasibility tolerance
    :valfun_ϵ => 1e-6,  # valfun tolerance in discarding rules, etc.
    :use_div => false,  # whether to use \ in computing valfun
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


# run_tabu_valfun_compare(G, w, sdp_params, qstab_params)
run_test_qstab_valfuns(G, w, qstab_params, sdp_params)