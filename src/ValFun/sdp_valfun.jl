include("valfun_utils.jl")

function solve_grotschel_dual(model, G, w)
    n = nv(G)
    i0 = n + 1
    E = edges(G)
    @variable(model, t)
    @variable(model, λ[1:n])
    @variable(model, Λ[1:length(E)])  # vector of lambda_ij
    # @variable(model, λ[1:n] >= 0)
    # @variable(model, Λ[1:length(E)] >= 0)

    # Sparse
    # Q = Symmetric(sparse([src(e) for e in E], [dst(e) for e in E], Λ, i0, i0) + sparse(collect(1:n), fill(i0, n), -λ, i0, i0)) * 0.5 + sparse(collect(1:n), collect(1:n), λ, i0, i0) + Diagonal([-w; t])
    
    # Symmetric{Sparse}
    Q = Symmetric(sparse([src(e) for e in E], [dst(e) for e in E], Λ, i0, i0) + sparse(collect(1:n), fill(i0, n), -λ, i0, i0)) * 0.5 + Diagonal([λ - w; t])

    # display(Q)
    @constraint(model, X, Q in PSDCone())
    @objective(model, Min, t)
    @time optimize!(model)

    X_val = Symmetric(dual.(X))
    Q_val = Symmetric(value.(Q))
    obj_val = value(t)

    return (X=X_val, Q=Q_val, value=obj_val)
end

function solve_grotschel_primal(model, G, w)
    n = nv(G)
    i0 = n + 1
    E_c = edges(complement(G))
    @variable(model, x[1:n])
    @variable(model, X[1:length(E_c)])  # off-diagonal entries
    # Sparse
    # X_plus = Symmetric(sparse([src(e) for e in E_c], [dst(e) for e in E_c], X, i0, i0) + sparse(collect(1:n), fill(i0, n), x, i0, i0)) + sparse(collect(1:n), collect(1:n), x, i0, i0)
    # X_plus[i0, i0] = 1

    # Symmetric{Sparse}
    X_plus = Symmetric(sparse([src(e) for e in E_c], [dst(e) for e in E_c], X, i0, i0) + sparse(collect(1:n), fill(i0, n), x, i0, i0)) + Diagonal([x; 1])

    # display(X_plus)
    @constraint(model, Q, X_plus in PSDCone())
    @objective(model, Max, w' * x)
    @time optimize!(model)

    X_val = Symmetric(value.(X_plus))
    Q_val = Symmetric(dual.(Q))
    obj_val = objective_value(model)

    return (X=X_val, Q=Q_val, value=obj_val)
end

function solve_lovasz_dual(model, E, w)
    n = nv(G)
    E = edges(G)
    @variable(model, t)
    @variable(model, λ[1:length(E)])

    sqrt_w = sqrt.(w)
    W = sqrt_w * transpose(sqrt_w)
    Q = Symmetric(sparse([src(e) for e in E], [dst(e) for e in E], λ, n, n) + Diagonal(fill(t, n)) - W)

    # display(Q)
    @constraint(model, X, Q in PSDCone())
    @objective(model, Min, t)
    @time optimize!(model)

    X_val = Symmetric(dual.(X))
    Q_val = Symmetric(value.(Q))
    obj_val = objective_value(model)

    return (X=X_val, Q=Q_val, value=obj_val)
end


function solve_lovasz_primal(model, E, w)
    n = nv(G)
    E_c = edges(complement(G))
    @variable(model, X_diag[1:n])
    @variable(model, X_off_diag[1:length(E_c)])

    X = Symmetric(sparse([src(e) for e in E_c], [dst(e) for e in E_c], X_off_diag, n, n)) + Diagonal(X_diag)

    # display(X)
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

# Solve dual SDP
function dualSDP(G, w, solve_dual=true, formulation="grotschel"; solver="SCS", ϵ=0, feas_ϵ=0, verbose=false)
    n = nv(G)
    i0 = n + 1
    E = collect(edges(G))
    model = Model()
    set_sdp_optimizer(model; solver=solver, ϵ=ϵ, feas_ϵ=feas_ϵ, verbose=verbose)

    if formulation == "grotschel"
        if solve_dual
            sol = solve_grotschel_dual(model, G, w)
        else
            sol = solve_grotschel_primal(model, G, w)
        end
    elseif formulation == "lovasz"
        if solve_dual
            sol = solve_lovasz_dual(model, G, w)
        else
            sol = solve_lovasz_primal(model, G, w)
        end
    end
    println(eigmin(Matrix(sol.Q)))
    return sol
end

# Value funciton approximation
# use value of the PSD matrix in the dual SDPpseudoinverse
function valfun(Q; ϵ=1e-8)
    n = size(Q,1) - 1
    i0 = n + 1
    A = Symmetric(Q[1:n,1:n])
    b = Q[1:n, i0]
    return (S, max_val) -> b[S]' * pinv(A[S,S], rtol=ϵ) * b[S]  # pinv probably takes a significant portion of time. The pinv of sparse matrix is often dense. Can we do something else?
end

function valfun_ls(Q)
    n = size(Q,1) - 1
    i0 = n + 1
    A = Symmetric(Q[1:n,1:n])
    b = Q[1:n, i0]
    return (S, max_val) -> b[S]' * (A[S, S] \ b[S])
end

# More accurate value function by explicitly solving SDP
# use value of the PSD matrix in the dual SDP, and solve an SDP on its submatrix
# to obtain a value function
function valfun_sdp_explicit(Q; solver="COPT", ϵ=0, feas_ϵ=0)
    n = size(Q,1) - 1
    i0 = n + 1
    A = Symmetric(Q[1:n,1:n])
    b = Q[1:n, i0]
    val = (S, max_val) -> begin
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

# More accurate value function by bisection
# solve the submatrix SDP by bisection
function valfun_bisect(Q; ϵ=1e-10, psd_ϵ=1e-8)
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
    n = size(Q, 1) - 1
    i0 = n + 1
    A = Symmetric(Q[1:n, 1:n])
    b = Q[1:n, i0]
    # isPSD = (t, S) -> eigmin([t b[S]'; b[S] A[S,S]]) > -psd_ϵ
    isPSD = (t, S) -> isposdef([t b[S]'; b[S] A[S,S]])
    return (S, max_val) -> bisection(S, isPSD, 0, max_val; ϵ=ϵ)
end
