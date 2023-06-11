using Combinatorics
using LinearAlgebra
using JuMP, SCS, MosekTools, COPT

include("graph_utils.jl")
include("valfun.jl")
include("valfun_lib.jl")

# 1. LP to find weights and val such that all vertices are not discarded, while
# minimizing val(N)
# 2. find θ value using the weights. θ should match val(N)
function find_bad_val(G)
    n = nv(G)
    N = collect(1:n)

    model = Model(optimizer_with_attributes(COPT.Optimizer, "Logging" => false))
    subsets = collect(powerset(N))
    @variable(model, v[subsets] >= 0)
    @variable(model, w[1:n] >= 1)
    for s in powerset(N, 1)
        for i in setdiff(N, s)
            @constraint(model, v[s] <= v[sort(vcat(s, [i]))])  # Lemma 2(1)
        end
    end
    for s in powerset(N, 1, floor(Int64, n/2))
        disconnected_vertices = del_set(G, N, s)  # the set of vertices disconnected to s
        for t in powerset(setdiff(N, s), 1)
            if issubset(t, disconnected_vertices)
                @constraint(model, v[s] + v[t] == v[sort(vcat(s, t))])  # Lemma 2(2)
            # else
            #     @constraint(model, v[s] + v[t] >= v[sort(vcat(s, t))])  # subadditivity
            end
        end
    end

    for i in N
        @constraint(model, v[N] - v[del_lp(G, N, i)] == w[i])  # discard rule
        @constraint(model, v[[i]] >= w[i])  # Lemma 2(3)
    end
    @constraint(model, v[Int64[]] == 0)
    @objective(model, Min, v[N])
    println("Optimizing...")
    optimize!(model)
    println(solution_summary(model))
    if termination_status(model) != MOI.OPTIMAL
        return nothing, nothing
    end

    bad_w = value.(w)
    bad_val = (S) -> value(v[S])
    println(bad_w)

    sdp_sol = dualSDP(G, bad_w; solver="Mosek", ϵ=1e-9)
    println("The theta value is ", sdp_sol.value, ", and val(N) is ", value(v[N]))

    return bad_w, bad_val, sdp_sol
end


function del_lp(G,S,i)
    s_del = setdiff(S, vcat(neighbors(G,i), [i]))
    if s_del == []
        s_del = Int64[]
    end
    return s_del
end

function del_set(G, S, vertices)
    s_del = setdiff(S, union([neighbors(G,i) for i in vertices]..., vertices))
    if s_del == []
        s_del = Int64[]
    end
    return s_del
end

#-----------------------------------------
use_complement = false
graph_name = "ivan-6-bad"
family = "chordal"
G = load_family_graph(graph_name, family, use_complement)
# G, graph_name = generate_family_graph("hole", 6, use_complement)

# plot_graph(G, graph_name, use_complement; add_label=true)
n = nv(G)
println(graph_name, " ", use_complement)
println(n)
println(ne(G))

bad_w, bad_val, sdp_sol = find_bad_val(G)
# print_valfun(bad_val, n)
θ = sdp_sol.value
val = valfun(Matrix(sdp_sol.Q))

# check if bad_val is good or bad (whether each vertex is in some max stable set)
# println("stable set test")
# stable_set_test(G, bad_w, θ, bad_val)
println("theta test")
theta_test(G, bad_w, θ, bad_val; solver="SCS")
