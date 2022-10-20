using Combinatorics
using LinearAlgebra
using JuMP, SCS, MosekTools

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

    model = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => false, "eps_rel" => 1e-7, "eps_abs" => 1e-7))
    subsets = collect(powerset(N))
    @variable(model, v[subsets] >= 0)
    @variable(model, w[1:n] >= 1)
    for s in subsets
        for i in s
            @constraint(model, v[s] - v[del_lp(G, s, i)] >= w[i])
        end
    end
    for i in N
        @constraint(model, v[N] - v[del_lp(G, N, i)] == w[i])
        @constraint(model, v[[i]] == w[i])
    end
    @objective(model, Min, v[N])

    optimize!(model)

    w_val = value.(w)
    println(w_val)
    θ = lovasz_sdp(G, value.(w))
    println("The theta value is ", θ, ", and val(N) is ", value(v[N]))

    # greedy_verify(N, G, value.(w))
    theta_verify(N, G, value.(w), (S) -> value(v[S]), θ, 1e-3)
    return w_val
end


function lovasz_sdp(G, w)
    n = nv(G)
    V = collect(vertices(G))
    W = [sqrt(w[i] * w[j]) for i in V, j in V]

    model = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => false, "eps_rel" => 1e-7, "eps_abs" => 1e-7))
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

#-----------------------------------------
using DelimitedFiles
use_complement = true
graph_name = "connecting-15-1.0-5"
family = "chordal"
G = load_family_graph(graph_name, family, use_complement)

# use_complement = false
# G, graph_name = generate_family_graph("hole", 6, use_complement)

n = nv(G)
println(n)
println(ne(G))

# w = ones(n)
# find_bad_val(G, w)

# find_bad_val_rand(G)

bad_w = find_bad_val_w(G)
val, θ = get_valfun(G, bad_w)
perfect_tabu_valfun_verify_I(G, bad_w, val, θ, 1e-3)
