using JuMP
using LinearAlgebra, SparseArrays
using Combinatorics
using Graphs

include("graph_utils.jl")


del = (G,S,i) -> setdiff(S, vcat(neighbors(G,i), [i]))
del! = (G,S,i) -> setdiff!(S, vcat(neighbors(G,i), [i]))


function print_valfun(val, n, max_size=n)
    subsets = powerset(1:n, 0, max_size)
    for s in subsets
        println(s, " ", val(s))
    end
end

# rounding based on value function, a heuristics for general graphs
function round_valfun(G, w, θ, val)
    n = nv(G)
    S = collect(1:n)
    x_stable = falses(n)

    while length(S) > 0
        idx = argmax(w[j] + val(del(G,S,j)) for j in S)
        j = S[idx]
        x_stable[j] = 1
        del!(G, S, j)
        println("Current weight: ", w' * x_stable)
    end
    return x_stable
end

# use value function to iteratively discard and pick vertices to form a stable
# set
function tabu_valfun(G, w, θ, val; max_rounds=nv(G), ϵ=1e-4, verbose=true)
    n = nv(G)
    S = collect(1:n)
    x_stable = falses(n)
    current_weight = 0
    vertex_value_discard!(w, val, S; ϵ=ϵ, verbose=verbose)
    for i in 1:max_rounds
        fixed_point_discard!(G, w, θ, val, S, current_weight; ϵ, verbose)
        if !isempty(S)
            pick_vertex!(G, S, x_stable)
            current_weight = w' * x_stable
            if verbose
                println("Current weight: ", current_weight)
                println("Remaining value: ", val(S))
            end
        else
            break
        end
    end
    return x_stable, S
end


# pick a vertex in S (update the boolean vector x_stable) and update S
function pick_vertex!(G, S, x_stable; v_to_pick=S[1])
    x_stable[v_to_pick] = true
    del!(G, S, v_to_pick)
end


function vertex_value_discard!(w, val, S; ϵ=1e-6, verbose=true)
    T = copy(S)
    for v in T
        val_v = val([v])
        if w[v] < val_v - ϵ
            setdiff!(S, v)
            if verbose
                println(v, " is discarded. Value: ", val_v)
            end
        end
    end
end


function set_value_discard!(G, w, θ, val, S, current_weight=0; ϵ=1e-4, verbose=true)
    T = copy(S)
    for v in T
        val_v_c = val(del(G, T, v))
        if w[v] + val_v_c < θ - current_weight - ϵ
            setdiff!(S, v)
            if verbose
                println(v, " is discarded. Value of LHS: ", w[v] + val_v_c, "; value of RHS: ", θ - current_weight)
            end
        end
    end
end

# repeatedly apply set_value_discard until no more vertices can be discarded
function fixed_point_discard!(G, w, θ, val, S, current_weight=0; ϵ=1e-4, verbose=true)
    prev_size = Inf
    n_iter = 0
    while length(S) < prev_size
        prev_size = length(S)
        set_value_discard!(G, w, θ, val, S, current_weight; ϵ, verbose)
        n_iter += 1
    end
    n_iter -= 1
    if verbose && n_iter > 0
        println("Fixed point discard complete after ", n_iter, " round(s). Remaining vertices: ", prev_size)
    end
    if n_iter > 1
        println("Warning! More than 1 iterations of discarding observed.")
    end
end

# apply tabu_valfun() to pick n_rounds number of vertices, then for each vertex
# in the remaining set, test whether it is in some maximum stable set
function tabu_valfun_test(G, w, θ, val; n_rounds=0, ϵ=1e-6, solver="SCS", solver_ϵ=1e-7, graph_name=nothing, use_complement=nothing, verbose=false)
    # first, pick a specified number of vertices with tabu_valfun()
    x_stable, S = tabu_valfun(G, w, θ, val; max_rounds=n_rounds, ϵ=ϵ, verbose=verbose)
    # discard bad vertices in the resulting set
    fixed_point_discard!(G, w, θ, val, S, w' * x_stable; ϵ=ϵ, verbose=verbose)

    # S = set_value_discard(G, w, θ, val, S, ϵ)
    # println("First round complete. Remaining size: ", length(S))
    # if graph_name != nothing && use_complement != nothing
    #     plot_graph_no_isolated(G[S], graph_name, use_complement)
    # end

    # stable_set_test(G, w, θ, val, S, x_stable; ϵ=ϵ, verbose=verbose)
    theta_test(G, w, θ, val, S, x_stable; ϵ=ϵ, solver=solver, solver_ϵ=solver_ϵ)
end


# verify each vertex in S is in some maximum stable set by picking it first and
# then iteratively discarding and picking (as in tabu_valfun)
function stable_set_test(G, w, θ, val, S=collect(1:nv(G)), x_stable=falses(nv(G)); ϵ=1e-4, verbose=false)
    for first_v in S
        T = copy(S)
        x = copy(x_stable)
        pick_vertex!(G, T, x; v_to_pick=first_v)
        current_weight = w' * x
        while true
            fixed_point_discard!(G, w, θ, val, T, current_weight; ϵ=ϵ, verbose=verbose)
            if !isempty(T)
                pick_vertex!(G, T, x)
                current_weight = w' * x
            else
                break
            end
        end
        println("Finished starting with ", string(first_v), ". Final weight: ", current_weight)
    end
end

# verify each vertex in S is in some maximum stable set by picking it and
# computing the theta value of the remaining subgraph
function theta_test(G, w, θ, val, S=collect(1:nv(G)), x_stable=falses(nv(G)); ϵ=1e-6, solver="SCS", solver_ϵ=1e-7)
    n = nv(G)
    for first_v in S
        T = copy(S)
        x = copy(x_stable)
        pick_vertex!(G, T, x; v_to_pick=first_v)
        current_weight = w' * x
        # compute theta on the subgraph G[T]
        sol = dualSDP(G[T], w[T]; solver=solver, ϵ=solver_ϵ, verbose=false)
        θ_T  = sol.value
        val_T = val(T)
        if abs(val_T - θ_T) > ϵ || abs(θ - θ_T) > w[first_v] + current_weight + ϵ
            println("Warning! Original theta: ", θ, ". When ", string(first_v), " with weight ", w[first_v], " is picked, current weight: ", current_weight, "; remaining value: ", val_T, "; remaining theta: ", θ_T)
        end
    end
end


# test the subadditivity of a list of value functions on the subset S
# the value functions should be ordered in terms of accuracy, with the more
# inaccurate (but efficient) in the front
function test_subadditivity(θ, S, valfuns...; ϵ=1e-4)
    n = length(S)
    for s in powerset(S, 1, floor(Int64, n/2))
        for t in powerset(setdiff(S, s), 1)
            test_sets_subadditivity(s, t, valfuns...; ϵ=ϵ)
        end
    end
end

function random_test_subadditivity(θ, S, valfuns...; n_iter=10000, ϵ=1e-4)
    n = length(S)
    for i in 1:n_iter
        s_size = rand(1:floor(Int64, n/2))
        s = rand(S, s_size)
        t_set = setdiff(S, s)
        t_size = rand(1:length(t_set))
        t = rand(t_set, t_size)
        test_sets_subadditivity(s, t, valfuns...; ϵ=ϵ)
    end
end

# test the subadditivity of a list of value functions on two sets
# the value functions should be ordered in terms of accuracy, with the more
# inaccurate (but efficient) in the front
function test_sets_subadditivity(s, t, valfuns...; ϵ=1e-4)
    st = sort(vcat(s, t))
    val_s = nothing
    val_t = nothing
    val_st = nothing
    violated = true

    for val in valfuns
        val_s = val(s)
        val_t = val(t)
        val_st = val(st)
        if(val_s + val_t >= val_st - ϵ)
            violated = false
            break
        end
    end

    if violated
        println("Warning! Value of ", s, " is ", val_s, ", value of ", t, " is ", val_t, "; value of union is ", val_st, ". The difference is ", val_st - val_s - val_t)
    end
    return !violated
end

# function verify_neighbor_property(G, val, w, ϵ=1e-3)
#     println("Verifying neighbor property...")
#     for i in 1:nv(G)
#         val_i_neighbors = val(sort(vcat([i], neighbors(G, i))))
#         val_neighbors = val(neighbors(G, i))
#         if(abs(val_i_neighbors - max(w[i], val_neighbors)) > ϵ)
#             println("Warning! Value of ", i, " and neighbors is ", val_i_neighbors, "; value of neighbors of ", i, " is ", val_neighbors, ", and weight of ", i, " is ", w[i])
#         end
#     end
# end
