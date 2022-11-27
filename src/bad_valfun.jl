using Combinatorics
using LinearAlgebra
using JuMP, SCS, MosekTools, COPT

include("graph_utils.jl")
include("valfun.jl")

# 1. LP to find weights and val such that all vertices are not discarded, while
# minimizing val(N)
# 2. find θ value using the weights. θ should match val(N)
function find_bad_val_w(G)
    n = nv(G)
    N = collect(1:n)

    model = Model(optimizer_with_attributes(COPT.Optimizer, "Logging" => false))
    subsets = collect(powerset(N))
    @variable(model, v[subsets] >= 0)
    @variable(model, w[1:n] >= 1)
    for s in powerset(N, 1)
        # for i in s
        #     @constraint(model, v[s] - v[del_lp(G, s, i)] >= w[i])
        # end
        for i in setdiff(N, s)
            @constraint(model, v[s] <= v[sort(vcat(s, [i]))])
        end
    end
    for s in powerset(N, 1, floor(Int64, n/2))
        disconnected_vertices = del_set(G, N, s)
        for t in powerset(setdiff(N, s), 1)
            if issubset(t, disconnected_vertices)
                @constraint(model, v[s] + v[t] == v[sort(vcat(s, t))])
            # else
            #     @constraint(model, v[s] + v[t] >= v[sort(vcat(s, t))])
            end
        end
    end

    for i in N
        @constraint(model, v[N] - v[del_lp(G, N, i)] == w[i])
        @constraint(model, v[[i]] >= w[i])
    end
    @constraint(model, v[Int64[]] == 0)
    @objective(model, Min, v[N])
    println("Optimizing...")
    optimize!(model)
    println(solution_summary(model))
    if termination_status(model) != MOI.OPTIMAL
        return nothing, nothing
    end

    w_val = value.(w)
    bad_val = (S) -> value(v[S])
    println(w_val)

    θ = dualSDP(collect(edges(G)), w_val).value
    println("The theta value is ", θ, ", and val(N) is ", value(v[N]))

    return bad_val, w_val, θ
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

# plot_graph(G, graph_name, use_complement)

n = nv(G)
println(n)
println(ne(G))

# w = ones(n)
# find_bad_val(G, w)

# find_bad_val_rand(G)

bad_val, bad_w, θ = find_bad_val_w(G)
# val, θ = get_valfun(G, bad_w)
# print_valfun(val, n)

# stable_set_test(G, bad_w, bad_val)
theta_test(G, bad_w, θ, bad_val; solver="SCS")

# verify_subadditivity(G, bad_val, 1e-6)
# verify_subadditivity(G, val, 1e-6)
