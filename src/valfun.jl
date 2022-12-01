using JuMP, SCS, COSMO, MosekTools
using LinearAlgebra, SparseArrays
using Combinatorics
using Graphs

include("opt_utils.jl")
include("graph_utils.jl")

# Solve dual SDP
function dualSDP(E, w; solver="SCS", ϵ=1e-7, verbose=false)
    n = length(w)
    i0 = n + 1
    model = theta_sdp_model(solver=solver, ϵ=ϵ, verbose=verbose)
    @variable(model, t)
    @variable(model, λ[1:n])
    @variable(model, Λ[1:length(E)])  # vector of lambda_ij
    Q = Symmetric(sparse([src(e) for e in E], [dst(e) for e in E], Λ, i0, i0) + sparse(collect(1:n), fill(i0, n), -λ, i0, i0)) * 0.5 + sparse(collect(1:n), collect(1:n), λ, i0, i0) - Diagonal([w; 0])
    Q[i0, i0] = t
    @constraint(model, X, Q in PSDCone())
    @objective(model, Min, t)
    optimize!(model)
    X_r = Symmetric(dual.(X))
    val_r = value(t)
    lam_r = value.(λ)
    Lam_r = value.(Λ)
    Q_r = Symmetric(value.(Q))
    # println(eigmin(Matrix(Q_r)))

    if verbose
        println(max(Lam_r...))
        println(all(Lam_r .>= -1e-3))
        println(Lam_r[Lam_r .< -1e-3])
    end
    return (X=X_r, Q=Q_r, value=val_r, lam=lam_r, Lam=Lam_r)
end

# Value funciton approximation
# use value of the PSD matrix in the dual SDP to obtain a value function
function valfun(Q; ϵ=1e-8)
    n = size(Q,1)-1
    i0 = n+1
    A = Symmetric(Q[1:n,1:n])
    b = Q[1:n, i0]
    return S -> b[S]' * pinv(A[S,S],rtol=ϵ) * b[S]  # pinv probably takes a significant portion of time. The pinv of sparse matrix is often dense. Can we do something else?
end

# More accurate value function via SDP
# use value of the PSD matrix in the dual SDP, and solve an SDP on its submatrix
# to obtain a value function
function valfun_sdp(Q; solver="SCS", ϵ=1e-7)
    n = size(Q,1)-1
    i0 = n+1
    A = Symmetric(Q[1:n,1:n])
    b = Q[1:n, i0]
    val = S -> begin
        model = theta_sdp_model(solver=solver, ϵ=ϵ)
        @variable(model, t)
        Q_I = vcat(hcat(t, b[S]'), hcat(b[S], A[S,S]))
        @constraint(model, Q_I in PSDCone())
        @objective(model, Min, t)
        optimize!(model)
        value(t)
    end
    return val
end

# More accurate value function via bisection
# solve the submatrix SDP via bisection
function valfun_bisect(Q, θ; psd_ϵ=1e-8, bisect_ϵ=1e-10)
    upperBound = θ
    n = size(Q, 1) - 1
    i0 = n + 1
    A = Symmetric(Q[1:n, 1:n])
    b = Q[1:n, i0]
    isPSD = (t, S) -> eigmin([t b[S]'; b[S] A[S,S]]) > -psd_ϵ
    return S -> bisection(S, isPSD, 0, upperBound; ϵ=bisect_ϵ)
end

function bisection(S, condition, t0, t1; ϵ=1e-10)
    t = (t0 + t1)/2
    if t1 - t0 < ϵ
        return t
    end
    if condition(t, S)
        bisection(S, condition, t0, t)
    else
        bisection(S, condition, t, t1)
    end
end


function print_valfun(val, n, max_size=n)
    subsets = powerset(1:n, 0, max_size)
    for s in subsets
        println(s, " ", val(s))
    end
end


del = (G,S,i) -> setdiff(S, vcat(neighbors(G,i), [i]))
del! = (G,S,i) -> setdiff!(S, vcat(neighbors(G,i), [i]))


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
function tabu_valfun_test(G, w, θ, val; n_rounds=0, solver="SCS", ϵ=1e-6, graph_name=nothing, use_complement=nothing, verbose=false)
    # first, pick a specified number of vertices with tabu_valfun()
    x_stable, S = tabu_valfun(G, w, θ, val; max_rounds=n_rounds, ϵ=ϵ, verbose=verbose)
    # discard bad vertices in the resulting set
    fixed_point_discard!(G, w, θ, val, S, w' * x_stable; ϵ=ϵ, verbose=verbose)

    # S = set_value_discard(G, w, θ, val, S, ϵ)
    # println("First round complete. Remaining size: ", length(S))
    # if graph_name != nothing && use_complement != nothing
    #     plot_graph_no_isolated(G[S], graph_name, use_complement)
    # end

    stable_set_test(G, w, val, S, x_stable; ϵ=ϵ, verbose=verbose)
    # theta_test(G, w, θ, val, S, x_stable; solver, ϵ)
end


# verify each vertex in S is in some maximum stable set by picking it first and
# then iteratively discarding and picking (as in tabu_valfun)
function stable_set_test(G, w, val, S=collect(1:nv(G)), x_stable=falses(nv(G)); ϵ=1e-4, verbose=false)
    for first_v in S
        T = copy(S)
        y_stable = copy(x_stable)
        pick_vertex!(G, T, y_stable; v_to_pick=first_v)
        current_weight = w' * y_stable
        while true
            fixed_point_discard!(G, w, θ, val, T, current_weight; ϵ=ϵ, verbose=verbose)
            if !isempty(T)
                pick_vertex!(G, T, y_stable)
                current_weight = w' * y_stable
            else
                break
            end
        end
        println("Finished starting with ", string(first_v), ". Final weight: ", current_weight)
    end
end

# verify each vertex in S is in some maximum stable set by picking it and
# computing the theta value of the remaining subgraph
function theta_test(G, w, θ, val, S=collect(1:nv(G)), x_stable=falses(nv(G)); solver="SCS", ϵ=1e-6)
    n = nv(G)
    for first_v in S
        T = copy(S)
        y_stable = copy(x_stable)
        pick_vertex!(G, T, y_stable; v_to_pick=first_v)
        current_weight = w' * y_stable
        # compute theta on the subgraph G[T]
        E = collect(edges(G[T]))
        sol = dualSDP(E, w[T]; solver=solver, verbose=false)
        θ_T  = sol.value
        val_T = val(T)
        if abs(val_T - θ_T) > ϵ || abs(θ - θ_T) > w[first_v] + current_weight + ϵ
            println("Warning! Original theta: ", θ, ". When ", string(first_v), " with weight ", w[first_v], " is picked, current weight: ", current_weight, "; remaining value: ", val_T, "; remaining theta: ", θ_T)
        end
    end
end


# test the subadditivity of a value function
function test_val_subadditivity(G, val; ϵ=1e-6)
    n = nv(G)
    N = 1:n
    for s in powerset(N, 1, floor(Int64, n/2))
        for t in powerset(setdiff(N, s), 1)
            val_s = val(s)
            val_t = val(t)
            val_st = val(sort(vcat(s, t)))
            if (val_s + val_t < val_st - ϵ)
                println("Warning! Value of ", s, " is ", val_s, ", value of ", t, " is ", val_t, "; value of union is ", val_st, ". The difference is ", val_st - val_s - val_t)
            end
        end
    end
end


# test the subadditivity of a dual solution Q, using different methods of
# computing value function
function test_subadditivity(Q, θ, S=collect(1:size(Q, 1) - 1); solver="SCS", solver_ϵ=1e-7, ϵ=1e-4, psd_ϵ=1e-8, bisect_ϵ=1e-10)
    val = valfun(Q)
    val_sdp = valfun_sdp(Q; solver=solver, ϵ=solver_ϵ)
    val_bisect = valfun_bisect(Q, θ; psd_ϵ=psd_ϵ, bisect_ϵ=bisect_ϵ)
    n = length(S)
    for s in powerset(S, 1, floor(Int64, n/2))
        for t in powerset(setdiff(S, s), 1)
            test_sets_subadditivity(s, t, val, val_sdp, val_bisect; ϵ=ϵ)
            # if !(test_sets_subadditivity(s, t, val, val_sdp, val_bisect; ϵ=ϵ))
            #     break
            # end
        end
    end
end

function random_test_subadditivity(Q, θ, S=collect(1:size(Q, 1) - 1); solver="SCS", solver_ϵ=1e-7, n_iter=10000, ϵ=1e-4, psd_ϵ=1e-8, bisect_ϵ=1e-10)
    val = valfun(Q)
    val_sdp = valfun_sdp(Q; solver=solver, ϵ=solver_ϵ)
    val_bisect = valfun_bisect(Q, θ; psd_ϵ=psd_ϵ, bisect_ϵ=bisect_ϵ)
    n = length(S)
    for i in 1:n_iter
        s_size = rand(1:floor(Int64, n/2))
        s = rand(S, s_size)
        t_set = setdiff(S, s)
        t_size = rand(1:length(t_set))
        t = rand(t_set, t_size)
        test_sets_subadditivity(s, t, val, val_sdp, val_bisect; ϵ=ϵ)
    end
end

# test the subadditivity of two sets, using different methods of computing value
# function
function test_sets_subadditivity(s, t, val, val_sdp, val_bisect; ϵ=1e-4)
    st = sort(vcat(s, t))
    if(val(s) + val(t) < val(st) - ϵ)
        if (val_sdp(s) + val_sdp(t) < val_sdp(st) - ϵ)
            val_s = val_bisect(s)
            val_t = val_bisect(t)
            val_st = val_bisect(st)
            if (val_s + val_t < val_st - ϵ)
                println("Warning! Value of ", s, " is ", val_s, ", value of ", t, " is ", val_t, "; value of union is ", val_st, ". The difference is ", val_st - val_s - val_t)
                return false
            end
        end
    end
    return true
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

function dualSDP_test(E, w; solver="SCS", ϵ=1e-7, verbose=false, t_bound=0, test_negative=false, test_nonnegativity=false)
    n = length(w)
    i0 = n + 1
    model = theta_sdp_model(solver=solver, ϵ=ϵ, verbose=verbose)
    @variable(model, t)
    @variable(model, λ[1:n])
    @variable(model, Λ[1:length(E)])  # vector of lambda_ij
    Q = Symmetric(sparse([src(e) for e in E], [dst(e) for e in E], Λ, i0, i0) + sparse(collect(1:n), fill(i0, n), -λ, i0, i0)) * 0.5 + sparse(collect(1:n), collect(1:n), λ, i0, i0) - Diagonal([w; 0])
    Q[i0, i0] = t
    @constraint(model, X, Q in PSDCone())
    if t_bound > 0
        @constraint(model, t >= t_bound)
    else
        @objective(model, Min, t)
    end

    if test_negative
        @constraint(model, Λ[1] <= -0.1)
        @constraint(model, Λ[2] <= -0.1)
    end

    if test_nonnegativity
        @constraint(model, Λ .>= 0)
    end

    optimize!(model)
    X_r = Symmetric(dual.(X))
    val_r = value(t)
    lam_r = value.(λ)
    Lam_r = value.(Λ)
    Q_r = Symmetric(value.(Q))

    println(max(Lam_r...))
    println(all(Lam_r .>= -1e-3))
    println(Lam_r[Lam_r .< -1e-3])
    return (X=X_r, Q=Q_r, value=val_r, lam=lam_r, Lam=Lam_r)
end
