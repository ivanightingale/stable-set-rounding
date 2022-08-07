using JuMP, MosekTools, CSDP
using LinearAlgebra, SparseArrays
using Combinatorics
using Graphs
using EzXML
using GraphIO.GraphML
using Random
using COSMO

# Heuristic for max indep set
function max_ind_set(G,N=1000)
    cmax = 0
    Smax = []
    for i in 1:N
        S = independent_set(G,MaximalIndependentSet())
        if cmax < length(S)
            cmax = length(S)
            Smax = S
        end
    end
    return Smax
end

# Solve dual SDP
function dualSDP(E,w)
    n = length(w)
    i0 = n+1
    # model = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))
    model = Model(optimizer_with_attributes(COSMO.Optimizer, "complete_dual" => true))
    @variable(model, t)
    @variable(model, λ[1:n])
    @variable(model, Λ[1:length(E)])  # vector of lambda_ij
    Q = Symmetric(sparse([src(e) for e in E],[dst(e) for e in E], Λ, i0, i0) +  + sparse(collect(1:n), fill(i0, n), -λ, i0, i0)) * 0.5 + sparse(collect(1:n), collect(1:n), λ, i0, i0) - Diagonal([w;0])
    Q[i0, i0] = t
    @constraint(model, X, Q in PSDCone())
    @objective(model, Min, t)
    optimize!(model)
    X_r = Symmetric(dual.(X))
    val_r = value(t)
    lam_r = value.(λ)
    Lam_r = value.(Λ)
    Q_r = Symmetric(value.(Q))
    return (X=X_r, Q=Q_r, value=val_r, lam=lam_r, Lam=Lam_r)
end

# Value funciton approximation
function valfun(Q)
    tol = 1e-6      # Warning: this tolerance might need to be tweaked
    n = size(Q,1)-1
    i0 = n+1
    A = Symmetric(Q[1:n,1:n])
    b = Q[1:n,i0]
    return S -> b[S]'*pinv(A[S,S],rtol=tol)*b[S]  # pinv probably takes a significant portion of time
end

# Rounding based on value function
function round_valfun(G,w,val)
    del = (G,S,i) -> setdiff(S,vcat(neighbors(G,i),[i]))
    n = length(w)
    S = collect(1:n)
    xr = falses(n)
    while length(S)>0
        idx = argmax(w[j] + val(del(G,S,j)) for j in S) # this part could be very slow
        j = S[idx]
        xr[j] = 1
        S = del(G,S,j)
        println(w'*xr)
    end
    return xr
end

#-----------------------------------------
G = loadgraph("dat/sanr200-0-7.graphml", GraphMLFormat())
n = nv(G)

# # Graph
E = collect(edges(G))

# Weight Vector
w = ones(n)

# Solve SDP
sol = dualSDP(E,w)
println("SDP Value: ", sol.value)
println("Eigvals: ", last(eigvals(sol.X),3))

# Value fun & rounding
val = valfun(Matrix(sol.Q))
xr = round_valfun(G,w,val)
println("Rounded Value: ", w'*xr)
