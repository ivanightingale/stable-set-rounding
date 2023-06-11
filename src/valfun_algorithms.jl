using .ValFun

del = (G,S,i) -> setdiff(S, vcat(neighbors(G,i), [i]))
del! = (G,S,i) -> setdiff!(S, vcat(neighbors(G,i), [i]))

# Heuristic for non-perfect graphs, the original rounding algorithm
function round_valfun(G, w, θ, val; S = collect(1:n), x_stable = falses(n))
    n = nv(G)
    current_weight = w' * x_stable
    while length(S) > 0
        idx = argmax(w[j] + val(del(G,S,j)) for j in S)
        j = S[idx]
        x_stable[j] = 1
        del!(G, S, j)
        current_weight = w' * x_stable
        println("Current weight: ", current_weight)
    end
    return x_stable
end

# Use value function to iteratively discard and pick vertices to form a stable set.
function tabu_valfun(G, w, θ, val, S=collect(1:nv(G)), x_stable=falses(nv(G)); max_iter=nv(G), ϵ=1e-6, verbose=true)
    if max_iter == 0
        return x_stable, S
    end
    n = nv(G)
    current_weight = w' * x_stable
    vertex_value_discard!(w, θ, val, S; ϵ=ϵ, verbose=verbose)
    for i in 1:max_iter
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

# Pick a vertex in S (default to the first vertex in S), then update the boolean vector x_stable and S
function pick_vertex!(G, S, x_stable; v_to_pick=S[1])
    x_stable[v_to_pick] = true
    del!(G, S, v_to_pick)
end

function vertex_value_discard!(w, θ, val, S; ϵ=1e-6, verbose=true)
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
    remaining_weight = θ - current_weight
    T = copy(S)
    for v in T
        val_v_c = val(del(G, T, v))
        if w[v] + val_v_c < remaining_weight - ϵ
            setdiff!(S, v)
            if verbose
                println(v, " is discarded. Value of LHS: ", w[v] + val_v_c, "; value of RHS: ", remaining_weight)
            end
        end
    end
end

# Repeatedly apply set_value_discard until no more vertices can be discarded
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

# Apply tabu_valfun() to pick n_start number of vertices. Next, discard vertices that should be
# discarded. Then, for each vertex in the remaining set, test whether it is in some maximum stable set
function tabu_valfun_test(G, w, θ, val; use_theta=false, n_start=0, ϵ=1e-6, solver="SCS", solver_ϵ=0, feas_ϵ=0, graph_name=nothing, use_complement=nothing, verbose=false)
    # First, pick a specified number of vertices with tabu_valfun(). If n_start=0, this will simply run
    # vertex_value_discard!()
    if n_start == 0
        S = collect(1:nv(G))
        x_stable = falses(nv(G))
        vertex_value_discard!(w, θ, val, S; ϵ=ϵ, verbose=verbose)
    else
        x_stable, S = tabu_valfun(G, w, θ, val; max_iter=n_start, ϵ=ϵ, verbose=verbose)
    end
    # Discard bad vertices in the resulting set.
    fixed_point_discard!(G, w, θ, val, S, w' * x_stable; ϵ=ϵ, verbose=verbose)

    if verbose
        println("Discarding complete. Testing starts...")
    end
    if isempty(S) && n_start == 0
        if verbose
            println("****** All vertices discarded, valfun test failed ******")
        end
        return false
    end
    if use_theta
        is_success = theta_test(G, w, θ, val, S, x_stable; ϵ=ϵ, solver=solver, solver_ϵ=solver_ϵ, feas_ϵ=feas_ϵ, verbose=verbose)
    else
        is_success = stable_set_test(G, w, θ, val, S, x_stable; ϵ=ϵ, verbose=verbose)
    end
    if verbose
        if is_success
            println("****** Valfun test passed ******")
        else
            println("****** Valfun test failed ******")
        end
    end
    return is_success
end

# Verify each vertex in S is in some maximum stable set by picking it first and
# then iteratively discarding and picking (as in tabu_valfun).
# Note that good points could also fail since the picking is naive. To double check, use theta_test().
function stable_set_test(G, w, θ, val, S=collect(1:nv(G)), initial_x_stable=falses(nv(G)); ϵ=1e-4, verbose=false)
    is_success = true
    for first_v in S
        if verbose
            println("**** Stable set test: vertex ", string(first_v), " ****")
        end
        T = copy(S)
        x = copy(initial_x_stable)
        pick_vertex!(G, T, x; v_to_pick=first_v)
        current_weight = w' * x
        # discard and pick vertices until exhausted
        while true
            fixed_point_discard!(G, w, θ, val, T, current_weight; ϵ=ϵ, verbose=verbose)
            if !isempty(T)
                if verbose
                    println("** Stable set test: picking vertex ", string(T[1]), " **")
                end
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

# Verify each vertex in S is in some maximum stable set by picking it and
# computing the theta value of the remaining subgraph
function theta_test(G, w, θ, val, S=collect(1:nv(G)), initial_x_stable=falses(nv(G)); ϵ=1e-6, solver="SCS", solver_ϵ=0, feas_ϵ=0, verbose=false)
    n = nv(G)
    is_success = true
    for first_v in S
        T = copy(S)
        x = copy(initial_x_stable)
        pick_vertex!(G, T, x; v_to_pick=first_v)
        current_weight = w' * x
        # compute theta on the subgraph G[T]
        sol = dualSDP(G[T], w[T]; solver=solver, ϵ=solver_ϵ, feas_ϵ=feas_ϵ, verbose=false)
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

# Compare operations performed by tabu_valfun_test() by two value functions on the same graph, and check
# whether they are the same.
function tabu_valfun_compare(G, w, θ, val_1, val_2; n_start=0, ϵ=1e-6, verbose=false)
    # First, pick n_start vertices
    if n_start == 0
        S_1 = collect(1:nv(G))
        S_2 = collect(1:nv(G))
        initial_x_1 = falses(nv(G))
        initial_x_2 = falses(nv(G))
        vertex_value_discard!(w, θ, val_1, S_1; ϵ, verbose)
        vertex_value_discard!(w, θ, val_2, S_2; ϵ, verbose)
    else
        initial_x_1, S_1 = tabu_valfun(G, w, θ, val_1; max_iter=n_start, ϵ, verbose)
        initial_x_2, S_2 = tabu_valfun(G, w, θ, val_2; max_iter=n_start, ϵ, verbose)
    end
    @assert S_1 == S_2
    S_1_temp = copy(S_1)
    S_2_temp = copy(S_2)
    current_weight = w' * initial_x_1
    fixed_point_discard!(G, w, θ, val_1, S_1, current_weight; ϵ, verbose)
    fixed_point_discard!(G, w, θ, val_2, S_2, current_weight; ϵ, verbose)
    # For perfect graphs, by our hypothesis, the above fixed_point_discard!() should have no effect.
    if n_start == 0
        @assert S_1 == S_1_temp
        @assert S_2 == S_2_temp
    end
    @assert S_1 == S_2
    # Execute tabu_valfun_test() simultaneously
    if verbose
        println("Discarding complete. Comparison starts...")
    end
    for first_v in S_1
        if verbose
            println("**** Stable set compare: vertex ", string(first_v), " ****")
        end
        T_1 = copy(S_1)
        T_2 = copy(S_2)
        x_1 = copy(initial_x_1)
        x_2 = copy(initial_x_2)
        pick_vertex!(G, T_1, x_1; v_to_pick=first_v)
        pick_vertex!(G, T_2, x_2; v_to_pick=first_v)
        @assert T_1 == T_2
        current_weight = w' * x_1
        # discard and pick vertices until exhausted
        while true
            fixed_point_discard!(G, w, θ, val_1, T_1, current_weight; ϵ=ϵ, verbose=verbose)
            fixed_point_discard!(G, w, θ, val_2, T_2, current_weight; ϵ=ϵ, verbose=false)
            @assert T_1 == T_2
            if !isempty(T_1)
                if verbose
                    println("** Stable set compare: picking vertex ", string(T_1[1]), " **")
                end
                pick_vertex!(G, T_1, x_1)
                pick_vertex!(G, T_2, x_2)
                current_weight = w' * x_1
            else
                break
            end
        end
        @assert current_weight >= θ - ϵ
    end
    println("****** Valfuns equivalent ******") 
end
