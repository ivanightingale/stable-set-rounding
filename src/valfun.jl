using JuMP, SCS, COSMO, MosekTools
using LinearAlgebra, SparseArrays
using Combinatorics
using Graphs

include("graph_utils.jl")

# Solve dual SDP
function dualSDP(E, w, solver, verbose=true)
    n = length(w)
    i0 = n + 1
    ϵ = 1e-7
    if solver == "COSMO"
        model = Model(optimizer_with_attributes(COSMO.Optimizer, "eps_abs" => ϵ, "eps_rel" => ϵ, "decompose" => true, "max_iter" => 1000000, "verbose" => verbose))  # for larger graphs
    elseif solver == "SCS"
        model = Model(optimizer_with_attributes(SCS.Optimizer, "eps_abs" => ϵ, "eps_rel" => ϵ, "max_iters" => 1000000, "verbose" => verbose))
    else
        model = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => !verbose))
    end
    @variable(model, t)
    @variable(model, λ[1:n])
    @variable(model, Λ[1:length(E)])  # vector of lambda_ij
    Q = Symmetric(sparse([src(e) for e in E],[dst(e) for e in E], Λ, i0, i0) + sparse(collect(1:n), fill(i0, n), -λ, i0, i0)) * 0.5 + sparse(collect(1:n), collect(1:n), λ, i0, i0) - Diagonal([w; 0])
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
    return (X=X_r, Q=Q_r, value=val_r, lam=lam_r, Lam=Lam_r)
end

# Value funciton approximation
function valfun(Q)
    tol = 1e-6
    n = size(Q,1)-1
    i0 = n+1
    # A = Symmetric(Q[1:n,1:n])
    A = Q[1:n,1:n]
    b = Q[1:n,i0]
    return S -> b[S]' * pinv(A[S,S],rtol=tol) * b[S]  # pinv probably takes a significant portion of time. The pinv of sparse matrix is often dense. Can we do something else?
    # TODO: solve the SDP explicitly
end

del = (G,S,i) -> setdiff(S, vcat(neighbors(G,i), [i]))

function get_valfun(G, w, solver="SCS", verbose=false)
    E = collect(edges(G))
    # @time sol = dualSDP(E, w, "SCS", false)
    sol = dualSDP(E, w, solver, verbose)
    println("SDP Value: ", sol.value)
    # println("Eigvals: ", last(eigvals(sol.X), 3))
    val = valfun(Matrix(sol.Q))  # convert SparseMatrix to Matrix
    return val, sol.value
end

# rounding based on value function, a heuristics for general graphs
function round_valfun(G, w, val, θ)
    n = length(w)
    S = collect(1:n)
    xr = falses(n)

    while length(S) > 0
        idx = argmax(w[j] + val(del(G,S,j)) for j in S)
        j = S[idx]
        xr[j] = 1
        S = del(G,S,j)
        println("Current weight: ", w' * xr)
    end
    return xr
end


# iteratively apply discard rules and pick the first candidate for imperfect graphs
# TODO: make it work for the remaining cases
function tabu_valfun_imperfect(G, w, val, θ, ϵ=1e-6)
    n = length(w)
    S = collect(1:n)
    xr = falses(n)
    current_weight = 0

    S = vertex_value_discard(S, val, w, ϵ)

    while !isempty(S)
        S = fixed_point_discard(S, G, θ, val, w, ϵ, current_weight)
        if !isempty(S)
            v = S[1]  # pick a vertex in the remaining candidates
            xr[v] = true
            S = del(G, S, v)  # this ensures the final output is a stable set
            current_weight = w' * xr
            println("Current weight: ", current_weight)
        end
    end
    return xr
end


# for perfect graphs, apply a single round of elimination
function tabu_valfun_perfect(G, w, val, θ, ϵ)
    n = length(w)
    S = collect(1:n)
    xr = falses(n)

    S = vertex_value_discard(S, val, w, ϵ)
    S = set_value_discard(S, G, θ, val, w, ϵ)
    while !isempty(S)
        v = S[1]  # pick a vertex in the remaining candidates
        xr[v] = true
        S = del(G, S, v)  # this ensures the final output is a stable set
        current_weight = w' * xr
        println("Current weight: ", current_weight)
    end
    return xr
end


# for perfect graphs, verify that applying a single round of elimination results in the
# candidate set I
function perfect_tabu_valfun_verify_I(G, w, val, θ, ϵ, graph_name=nothing, use_complement=nothing)
    n = length(w)
    S = collect(1:n)
    xr = falses(n)

    S = vertex_value_discard(S, val, w, ϵ)
    S = set_value_discard(S, G, θ, val, w, ϵ)
    println("Discard complete. Remaining size: ", length(S))

    # if graph_name != nothing && use_complement != nothing
    #     plot_graph_no_isolated(G[S], graph_name, use_complement)
    # end

    # greedy_verify(S, G, w)
    theta_verify(S, G, w, val, θ, ϵ)
end


# function primal_discard(S, X, ϵ=1e-6)
#     return filter(n -> X[end, n] > ϵ, S)
# end


# for each vertex in S, pick it and then greedily pick subsequent vertices, and check the weight of the resulting stable set
function greedy_verify(S, G, w)
    n = length(w)
    xr = falses(n)
    T = copy(S)
    for first_v in T
        xr[first_v] = true
        S = del(G, S, first_v)
        while !isempty(S)
            v = S[1]  # pick a vertex in the remaining candidates
            xr[v] = true
            S = del(G, S, v)  # this ensures the final output is a stable set
        end
        println("Finished starting with ", string(first_v), ". Final weight: ", w' * xr)
        xr = falses(n)
        S = copy(T)
    end
end

# for each vertex in S, pick it and explicitly compute the theta value of the remaining subgraph
function theta_verify(S, G, w, val, θ, ϵ)
    n = length(w)
    xr = falses(n)
    T = copy(S)
    for first_v in T
        xr[first_v] = true
        S = del(G, S, first_v)
        E = collect(edges(G[S]))
        sol = dualSDP(E, w[S], "SCS", false)
        val_S = val(S)
        println("Original theta: ", θ, ". Value function when ", string(first_v), " with weight ", w[first_v], " is picked: ", val_S, "; new theta value: ", sol.value)
        if abs(val_S - sol.value) > ϵ || abs(θ - sol.value) > w[first_v] + ϵ
            println("Warning!")
        end
        S = copy(T)
    end
end

# pick vertices in S in a specified order, and compute the weight of the resulting stable set
function manual_verify(S, G, w, order=[])
    n = length(w)
    xr = falses(n)
    for v in order
        if isempty(S)
            break
        end
        if !(v in S)
            println("Warning: vertex ", v, " is not in the set. Using S[1] instead.")
            v = S[1]
        end
        xr[v] = true
        S = del(G, S, v)
    end
    while !isempty(S)
        v = S[1]
        xr[v] = true
        S = del(G, S, v)
    end
    println("Finished with picking order ", order, " Final weight: ", w' * xr)
end


function vertex_value_discard(S, val, w, ϵ, verbose=true)
    T = copy(S)
    for v in T
        val_v = val([v])
        if w[v] < val_v - ϵ
            S = setdiff(S, v)
            if verbose
                println(v, " is discarded. Value: ", val_v)
            end
        end
    end
    return S
end


function set_value_discard(S, G, θ, val, w, ϵ, current_weight=0, verbose=true)
    T = copy(S)
    for v in T
        val_v_c = val(del(G, T, v))
        if w[v] + val_v_c < θ - current_weight - ϵ
            S = setdiff(S, v)
            if verbose
                println(v, " is discarded. Value of LHS: ", w[v] + val_v_c, "; value of RHS: ", θ - current_weight)
            end
        end
    end
    return S
end


function fixed_point_discard(S, G, θ, val, w, ϵ, current_weight=0, verbose=true)
    terminate = false
    current_size = length(S)
    iter = 1
    while !terminate
        S = set_value_discard(S, G, θ, val, w, ϵ, current_weight, verbose)
        if length(S) == current_size
            terminate = true
        end
        current_size = length(S)
        println("Fixed point ", iter, " complete. Remaining vertices: ", current_size)
        iter += 1
    end
    return S
end
