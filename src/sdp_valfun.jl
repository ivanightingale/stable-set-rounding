include("valfun_utils.jl")

# Solve dual SDP
function dualSDP(G, w; solver="SCS", ϵ=0, feas_ϵ=0, verbose=false)
    n = nv(G)
    i0 = n + 1
    E = collect(edges(G))
    model = Model()
    set_sdp_optimizer(model; solver=solver, ϵ=ϵ, feas_ϵ=feas_ϵ, verbose=verbose)
    @variable(model, t)

    @variable(model, λ[1:n])
    @variable(model, Λ[1:length(E)])  # vector of lambda_ij
    # @variable(model, λ[1:n] >= 0)
    # @variable(model, Λ[1:length(E)] >= 0)

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

    # if verbose
    #     println(max(Lam_r...))
    #     println(all(Lam_r .>= -1e-3))
    #     println(Lam_r[Lam_r .< -1e-3])
    # end
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
function valfun_sdp_explicit(Q; solver="SCS", ϵ=0, feas_ϵ=0)
    n = size(Q,1)-1
    i0 = n+1
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