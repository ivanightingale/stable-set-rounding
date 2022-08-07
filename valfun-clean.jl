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

# Random connected graph
function rand_conn_graph(n,p)
    while true
        G = erdos_renyi(n,p)
        if is_connected(G) # && is_bipartite(G)
            println("graph found")
            return G
        end
    end
end

# Construct PSD matrix of dual SDP
function getQ(E,t,lam,Lam)
    n = length(lam)
    i0 = n + 1
    Ik = collect(I(i0))
    M = (i,j) -> .5*(Ik[:,i]*Ik[:,j]' + Ik[:,j]*Ik[:,i]')
    Q0 = t * M(i0,i0)
    Q1 = sum(lam[i] * (M(i,i) - M(i,i0)) for i in 1:n)
    Q2 = sum(Lam[src(e),dst(e)] * M(src(e),dst(e)) for e in E)  # bottleneck
    println(typeof(Q0))
    println(typeof(Q1))
    println(typeof(Q2))
    return Symmetric(Q0 + Q1 + Q2 - Diagonal([w;0]))
end

# Solve dual SDP
function dualSDP(E,w)
    n = length(w)
    i0 = n+1
    # model = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))
    model = Model(optimizer_with_attributes(COSMO.Optimizer, "complete_dual" => true))
    @variable(model, t)
    @variable(model, lam[1:n])
    @variable(model, LamVec[1:length(E)])  # vector of lambda_ij
    Lam = sparse([src(e) for e in E],[dst(e) for e in E],LamVec,n,n)  # sparse matrix of lambda_ij
    println("Start getting Q")
    Q = getQ(E,t,lam,Lam)  # takes a lot of time
    println("end getting Q")
    @constraint(model, X, Q in PSDCone())
    @objective(model, Min, t)
    println("Optimizer starts")
    optimize!(model)
    X_r = Symmetric(dual.(X))
    val_r = value(t)
    lam_r = value.(lam)
    Lam_r = value.(Lam)
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
    return S -> b[S]'*pinv(A[S,S],rtol=tol)*b[S]
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
    end
    return xr
end

#-----------------------------------------
G = loadgraph("dat/hamming6-4.graphml", GraphMLFormat())
n = nv(G)

# n = 50                          # num vertices
# i0 = n+1                        # homogenizing index
# sd = Int(ceil(rand()*1000))     # random seed
# println("seed: ", sd)
# Random.seed!(sd)

# # Graph
# G = rand_conn_graph(n,.6)
E = collect(edges(G))

# Weight Vector
w = ones(n)

# Solve SDP
sol = dualSDP(E,w)
println("SDP Value: ", sol.value)
println("Eigvals: ", last(eigvals(sol.X),3))

# Value fun & rounding
val = valfun(sol.Q)
xr = round_valfun(G,w,val)
println("Rounded Value: ", w'*xr)
