using JuMP, MosekTools, SCS, COSMO, COPT
using LinearAlgebra, SparseArrays
using Combinatorics
using Graphs

###########################
# SDP-based value functions
###########################

function theta_sdp_model(;solver="SCS", ϵ=1e-7, verbose=false)
    if solver == "COSMO"
        model = Model(optimizer_with_attributes(COSMO.Optimizer, "eps_abs" => ϵ, "eps_rel" => ϵ, "decompose" => true, "max_iter" => 1000000, "verbose" => verbose))  # for larger graphs
    elseif solver == "SCS"
        model = Model(optimizer_with_attributes(SCS.Optimizer, "eps_abs" => ϵ, "eps_rel" => ϵ, "max_iters" => 1000000, "verbose" => verbose))
    elseif solver == "Mosek"
        model = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => !verbose, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => ϵ, "MSK_DPAR_INTPNT_CO_TOL_MU_RED" => ϵ))
    elseif solver == "COPT"
        model = Model(optimizer_with_attributes(COPT.ConeOptimizer, "Logging" => verbose))
    end
end

# Solve dual SDP
function dualSDP(G, w; solver="SCS", ϵ=1e-7, verbose=false)
    n = nv(G)
    i0 = n + 1
    E = collect(edges(G))
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
# use value of the PSD matrix in the dual SDPpseudoinverse
function valfun(Q; ϵ=1e-8)
    n = size(Q,1)-1
    i0 = n+1
    A = Symmetric(Q[1:n,1:n])
    b = Q[1:n, i0]
    return S -> b[S]' * pinv(A[S,S],rtol=ϵ) * b[S]  # pinv probably takes a significant portion of time. The pinv of sparse matrix is often dense. Can we do something else?
end

# More accurate value function by explicitly solving SDP
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

# More accurate value function by bisection
# solve the submatrix SDP by bisection
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


#######################################
# Clique polytope-based value functions
#######################################


# Generate a value function from the optimal dual variables
function valfun_qstab(λ, cliques)
    return S -> sum([λ[i] for (i, c) in enumerate(cliques) if !isempty(S ∩ c)])
end


# Find max stable set by starting with the fractional stable set polytope (edge
# polytope) and adding clique cutting planes
# Return the optimal dual variables indexed by cliques
function clique_stable_set_lp(G, w; verbose=false)
    n = nv(G)
    E = collect(edges(G))
    model = Model(optimizer_with_attributes(COPT.Optimizer, "Logging" => false, "LogToConsole" => false))
    @variable(model, x[1:n] >= 0)

    # cons = @constraint(model, [i in 1:n], x[i] <= 1)
    # cliques = [[i] for i in 1:n]
    cons = Vector{ConstraintRef}(undef, 0)
    cliques = Vector{Vector{Int64}}(undef, 0)
    for e in edges(G)
        push!(cons, @constraint(model, x[src(e)] + x[dst(e)] <= 1))
        push!(cliques, [src(e), dst(e)])
    end
    @objective(model, Max, w' * x)
    sub_sol_value = Inf
    while (sub_sol_value > 1)
        optimize!(model)
        sub_sol = max_clique(G, value.(x))
        sub_sol_value = sub_sol.value
        println(sub_sol_value)
        if sub_sol_value > 1
            clique = findall(sub_sol.z .> 0.5)
            push!(cliques, clique)
            push!(cons, @constraint(model, sum(x[clique]) <= 1))
        end
    end
    # println(objective_value(model))
    # println(cons)
    println(dual.(cons))
    println(cliques)
    return (x=value.(x), value=objective_value(model), λ=-dual.(cons), cliques=cliques)
end

# Solve an IP to find a max clique on G given weight w
# Return the optimal solution
# TODO: make sure the returned optimal solution actually exactly corresponds to a single clique
function max_clique(G, w)
    n = nv(G)
    model = Model(optimizer_with_attributes(COPT.Optimizer, "Logging" => false, "LogToConsole" => false))
    @variable(model, z[1:n], Bin)
    @constraint(model, [e in edges(complement(G))], z[src(e)] + z[dst(e)] <= 1)
    @objective(model, Max, w' * z)
    optimize!(model)
    return (z=value.(z), value=objective_value(model))
end

# Check whether a sorted list of vertices S form a clique in G
function is_clique(G, S)
    V = vertices(G)
    num_S = length(S)
    for i in 1:num_S
        v_i = S[i]
        neighbors_v_i = neighbors(G, v_i)
        for j in i+1:num_S
            v_j = S[j]
            if !(v_j in neighbors_v_i)
                println(v_j, " is not neighbor of ", v_i)
                return false
            end
        end
    end
    return true
end


# FIXME: the conversion formula seems incorrect
function qstab_to_sdp(G, w, λ, cliques)
    n = nv(G)
    i0 = n + 1
    E = collect(edges(G))
    val = valfun_qstab(λ, cliques)
    v = [val(i) for i in 1:n]
    Q = sparse(collect(1:n), fill(i0, n), -v, i0, i0) + sparse(collect(1:n), collect(1:n), v, i0, i0) * 2 - Diagonal([w; 0])
    for e in E
        i = src(e)
        j = dst(e)
        Q[i, j] = sum([λ[k] for (k, c) in enumerate(cliques) if(i in c && j in c)])
    end
    Q[i0, i0] = 0
    Q = Symmetric(Q)
    display(Q)
    return Matrix(Q)
end





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
