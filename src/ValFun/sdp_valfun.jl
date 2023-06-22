export get_valfun, valfun

# Get an SDP value function by solving either the primal or the dual of either the Lovasz or 
# the Grotschel formulation of the Lovasz theta SDP.
# COPT solves the primal faster than dual if G is dense, and solves the dual faster than primal
# if G is sparse.
function get_valfun(G, w, solve_dual=true, formulation=:grotschel; solver=:COPT, ϵ=0, feas_ϵ=0, use_div=true, pinv_rtol=1e-9, verbose=false)
    n = nv(G)
    i0 = n + 1
    E = collect(edges(G))
    model = Model()
    set_sdp_optimizer(model; solver=solver, ϵ=ϵ, feas_ϵ=feas_ϵ, verbose=verbose)

    if formulation == :grotschel
        sol = solve_grotschel_sdp(Val(solve_dual), model, G, w)
    elseif formulation == :lovasz
        sol = solve_lovasz_sdp(Val(solve_dual), model, G, w)
    end
    # println(eigmin(Matrix(sol.Q)))

    return (val=valfun(sol.Q; use_div, pinv_rtol), sol=sol)
end

# Grotschel dual
function solve_grotschel_sdp(::Val{true}, model, G, w)
    n = nv(G)
    i0 = n + 1
    E = edges(G)
    @variable(model, t)
    @variable(model, λ[1:n])
    @variable(model, Λ[1:length(E)])  # vector of lambda_ij
    Q = Symmetric(sparse([src(e) for e in E], [dst(e) for e in E], Λ, i0, i0) + sparse(collect(1:n), fill(i0, n), -λ, i0, i0)) * 0.5 + Diagonal([λ - w; t])

    @constraint(model, X, Q in PSDCone())
    @objective(model, Min, t)
    @time optimize!(model)

    X_val = Symmetric(dual.(X))
    Q_val = Symmetric(value.(Q))
    obj_val = value(t)

    return (X=X_val, Q=Q_val, value=obj_val)
end

# Grotschel primal
function solve_grotschel_sdp(::Val{false}, model, G, w)
    n = nv(G)
    i0 = n + 1
    E_c = edges(complement(G))
    @variable(model, x[1:n])
    @variable(model, X[1:length(E_c)])  # off-diagonal entries
    X_plus = Symmetric(sparse([src(e) for e in E_c], [dst(e) for e in E_c], X, i0, i0) + sparse(collect(1:n), fill(i0, n), x, i0, i0)) + Diagonal([x; 1])

    @constraint(model, Q, X_plus in PSDCone())
    @objective(model, Max, w' * x)
    @time optimize!(model)

    X_val = Symmetric(value.(X_plus))
    Q_val = Symmetric(dual.(Q))
    obj_val = objective_value(model)

    return (X=X_val, Q=Q_val, value=obj_val)
end

# Lovasz dual
function solve_lovasz_sdp(::Val{true}, model, G, w)
    n = nv(G)
    E = edges(G)
    @variable(model, t)
    @variable(model, λ[1:length(E)])
    sqrt_w = sqrt.(w)
    W = sqrt_w * transpose(sqrt_w)
    Q = Symmetric(sparse([src(e) for e in E], [dst(e) for e in E], λ, n, n) + Diagonal(fill(t, n)) - W)

    @constraint(model, X, Q in PSDCone())
    @objective(model, Min, t)
    @time optimize!(model)

    X_val = Symmetric(dual.(X))
    Q_val = Symmetric(value.(Q))
    obj_val = objective_value(model)

    return (X=X_val, Q=Q_val, value=obj_val)
end

# Lovasz primal
function solve_lovasz_sdp(::Val{false}, model, G, w)
    n = nv(G)
    E_c = edges(complement(G))
    @variable(model, X_diag[1:n])
    @variable(model, X_off_diag[1:length(E_c)])
    X = Symmetric(sparse([src(e) for e in E_c], [dst(e) for e in E_c], X_off_diag, n, n)) + Diagonal(X_diag)

    @constraint(model, tr(X) == 1)
    @constraint(model, Q, X in PSDCone())
    sqrt_w = sqrt.(w)
    W = sqrt_w * transpose(sqrt_w)
    @objective(model, Max, LinearAlgebra.dot(W, X))
    @time optimize!(model)

    X_val = Symmetric(value.(X))
    Q_val = Symmetric(dual.(Q))
    obj_val = objective_value(model)

    return (X=X_val, Q=Q_val, value=obj_val)
end

# Value funciton by psuedoinverse
function valfun(Q; use_div=true, pinv_rtol=1e-9)
    n = size(Q,1) - 1
    i0 = n + 1
    A = Symmetric(Q[1:n,1:n])
    b = Q[1:n, i0]
    # TODO: try using HSL
    if use_div
        return S -> b[S]' * (A[S, S] \ b[S])
    else
        return S -> b[S]' * pinv(A[S,S], rtol=pinv_rtol) * b[S]
    end
end

# Value function by explicitly solving SDP on submatrix
function valfun_explicit(Q; solver=:COPT, ϵ=0, feas_ϵ=0)
    n = size(Q,1) - 1
    i0 = n + 1
    A = Symmetric(Q[1:n,1:n])
    b = Q[1:n, i0]
    val = S -> begin
        model = Model()
        set_sdp_optimizer(model; solver=solver, ϵ=ϵ, feas_ϵ=feas_ϵ, verbose=false)
        @variable(model, t)
        Q_I = vcat(hcat(t, b[S]'), hcat(b[S], A[S,S]))
        @constraint(model, Q_I in PSDCone())
        @objective(model, Min, t)
        optimize!(model)
        value(t)
    end
    return val
end
