using Combinatorics
using LinearAlgebra
using JuMP, SCS, MosekTools, COPT

include("graph_utils.jl")
include("valfun.jl")

# for a perfect graph where not all vertices are in I, try to find a value function that is bad
function find_bad_val(G, w, θ=nothing)
    n = nv(G)
    N = collect(1:n)
    if θ == nothing
        θ = lovasz_sdp(G, w)
        println("The theta value is ", θ)
    end

    model = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => true, "eps_rel" => 1e-7, "eps_abs" => 1e-7))
    subsets = collect(powerset(N))
    @variable(model, v[subsets] >= 0)
    for s in subsets
        for i in s
            @constraint(model, v[s] - v[del_lp(G, s, i)] >= w[i])
        end
    end
    @constraint(model, v[N] == θ)
    for i in N
        @constraint(model, v[N] - v[del_lp(G, N, i)] == w[i])
        @constraint(model, v[[i]] == w[i])
    end

    optimize!(model)
end

# for a perfect graph, choose a random weight, and find value function. If any vertex
# is discarded, apply find_bad_val()
function find_bad_val_rand(G)
    n = nv(G)
    w = rand(n)
    println(w)
    val, θ = get_valfun(G, w)
    S = collect(1:n)
    T = copy(S)
    ϵ = 1e-3
    S = vertex_value_discard(S, val, w, ϵ)
    S = set_value_discard(S, G, θ, val, w, ϵ)
    discarded = setdiff(T, S)
    if isempty(discarded)
        println("No vertices are discarded. Exiting...")
    else
        find_bad_val(G, w, θ)
    end
end

# 1. SDP to find weights and val such that all vertices are not discarded, while minimizing
# val(N)
# 2. find θ value using the weights. θ should match val(N)
# 3. if such a val is found, verify all vertices are in I
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
    # print_valfun(bad_val, n)
    θ = lovasz_sdp(G, value.(w))
    println("The theta value is ", θ, ", and val(N) is ", value(v[N]))

    # greedy_verify(N, G, value.(w))
    theta_verify(N, G, value.(w), bad_val, θ, 1e-3, "SCS")
    return bad_val, w_val
end


function lovasz_sdp(G, w)
    n = nv(G)
    V = collect(vertices(G))
    W = [sqrt(w[i] * w[j]) for i in V, j in V]

    model = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => false, "eps_rel" => 1e-7, "eps_abs" => 1e-7))
    # model = Model(optimizer_with_attributes(COPT.Optimizer, "Logging" => false))
    @variable(model, Z[1:n, 1:n]);
    @constraint(model, Z ∈ PSDCone());
    @constraint(model, tr(Z) == 1.);
    for e in edges(G)
        @constraint(model, Z[src(e), dst(e)] == 0);
    end
    @objective(model, Max, dot(W, Z));

    optimize!(model)
    return value(objective_function(model))
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
graph_name = "connecting-15-1.0-1"
family = "chordal"
G = load_family_graph(graph_name, family, use_complement)

# plot_graph(G, graph_name, use_complement)

n = nv(G)
println(n)
println(ne(G))

# w = ones(n)
# find_bad_val(G, w)

# find_bad_val_rand(G)

bad_val, bad_w = find_bad_val_w(G)
# val, θ = get_valfun(G, bad_w)
# print_valfun(val, n)
# perfect_tabu_valfun_verify_I(G, bad_w, val, θ, 1e-3)

# verify_subadditivity(G, bad_val, 1e-6)
# verify_subadditivity(G, val, 1e-6)

# verify_neighbor_property(G, bad_val, bad_w, 1e-4)
# verify_neighbor_property(G, val, bad_w, 1e-4)
