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
                println(v, " is discarded. Value: ", val_v, "; weight: ", w[v])
            end
        end
    end
    if verbose
        println("Vertex value discard complete. Discarded ", length(T) - length(S), " vertices. Remaining vertices: ", length(S))
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
    original_size = length(S)
    prev_size = Inf
    n_iter = 0
    while length(S) < prev_size  # check whether fixed point has not been reached
        prev_size = length(S)
        set_value_discard!(G, w, θ, val, S, current_weight; ϵ, verbose)
        n_iter += 1
    end
    n_iter -= 1
    if verbose && n_iter > 0
        n_discarded = original_size - prev_size
        println("Fixed point discard complete after ", n_iter, " round(s). Discarded ", n_discarded, " vertices. Remaining vertices: ", prev_size)
        if n_iter > 1
            println("Warning! More than 1 iterations of discarding observed.")
        end
    end 
end

# apply tabu_valfun() to pick n_rounds number of vertices, then discard vertices that should be discarded;
# after that, for each vertex in the remaining set, test whether it is in some maximum stable set
function tabu_valfun_test(G, w, θ, val; use_theta=false, n_rounds=0, ϵ=1e-6, solver="SCS", solver_ϵ=1e-7, graph_name=nothing, use_complement=nothing, verbose=false)
    # First, pick a specified number of vertices with tabu_valfun().
    x_stable, S = tabu_valfun(G, w, θ, val; max_rounds=n_rounds, ϵ=ϵ, verbose=verbose)
    # Discard bad vertices in the resulting set.
    fixed_point_discard!(G, w, θ, val, S, w' * x_stable; ϵ=ϵ, verbose=verbose)

    if verbose
        println("Discarding complete. Testing starts...")
    end

    is_success = nothing
    if use_theta
        is_success = theta_test(G, w, θ, val, S, x_stable; ϵ=ϵ, solver=solver, solver_ϵ=solver_ϵ, verbose=verbose)
    else
        is_success = stable_set_test(G, w, θ, val, S, x_stable; ϵ=ϵ, verbose=verbose)
    end
    if verbose
        if is_success
            println("valfun test passed")
        else
            println("valfun test failed")
        end
    end
    return is_success
end


# verify each vertex in S is in some maximum stable set by picking it first and
# then iteratively discarding and picking (as in tabu_valfun)
function stable_set_test(G, w, θ, val, S=collect(1:nv(G)), initial_x_stable=falses(nv(G)); ϵ=1e-4, verbose=false)
    is_success = true
    for first_v in S
        T = copy(S)
        x = copy(initial_x_stable)
        pick_vertex!(G, T, x; v_to_pick=first_v)
        current_weight = w' * x
        # discard and pick vertices until exhausted
        while true
            fixed_point_discard!(G, w, θ, val, T, current_weight; ϵ=ϵ, verbose=verbose)
            if !isempty(T)
                pick_vertex!(G, T, x)
                current_weight = w' * x
            else
                break
            end
        end
        if current_weight < θ - ϵ
            is_success = false
            if verbose
               println("Warning! When picking vertices starting with ", string(first_v), ", final weight is ", current_weight, "; original theta: ", θ)
           end
        # elseif verbose
        #     println("When picking vertices starting with ", string(first_v), ", final weight is ", current_weight)
        end
    end
    return is_success
end

# verify each vertex in S is in some maximum stable set by picking it and
# computing the theta value of the remaining subgraph
function theta_test(G, w, θ, val, S=collect(1:nv(G)), x_stable=falses(nv(G)); ϵ=1e-6, solver="SCS", solver_ϵ=1e-7, verbose=false)
    n = nv(G)
    is_success = true
    for first_v in S
        T = copy(S)
        x = copy(x_stable)
        pick_vertex!(G, T, x; v_to_pick=first_v)
        current_weight = w' * x
        # compute theta on the subgraph G[T]
        sol = dualSDP(G[T], w[T]; solver=solver, ϵ=solver_ϵ, verbose=verbose)
        θ_T  = sol.value
        val_T = val(T)
        if abs(val_T - θ_T) > ϵ || abs(θ - θ_T) > w[first_v] + current_weight + ϵ
            is_success = false
            if verbose
                println("Warning! When ", string(first_v), " with weight ", w[first_v], " is picked, currently selected weight: ", current_weight, "; remaining value: ", val_T, "; remaining theta: ", θ_T, "; original theta: ", θ)
            end
        # elseif verbose
        #     println("When ", string(first_v), " with weight ", w[first_v], " is picked, remaining value: ", val_T, "; remaining theta: ", θ_T)
        end
    end
    return is_success
end


# Run a tabu_valfun_test for each dual solutions in λ_ext_points, and count the number of points that failed.
function test_qstab_valfuns(G, w, θ, λ_ext_points, cliques; use_theta=false)
    failure_count = 0
    for (i, λ) in enumerate(λ_ext_points)
        val_qstab = valfun_qstab(λ, cliques)
        val_qstab_sdp = valfun(qstab_to_sdp(G, w, λ, cliques))
        if !tabu_valfun_test(G, w, θ, val_qstab; use_theta=use_theta, ϵ=1e-6, solver="Mosek", solver_ϵ=1e-9, verbose=false) || !tabu_valfun_test(G, w, θ, val_qstab_sdp; use_theta=use_theta, ϵ=1e-6, solver="Mosek", solver_ϵ=1e-9, verbose=false)
            failure_count += 1
        end
    end
    return failure_count
end
