using JuMP, SCS, COSMO, MosekTools
using LinearAlgebra, SparseArrays
using Combinatorics
using MatrixMarket
using Graphs
using EzXML
using GraphIO.GraphML

# Solve dual SDP
function dualSDP(E,w)
    n = length(w)
    i0 = n + 1
    ϵ = 1e-7
    model = Model(optimizer_with_attributes(COSMO.Optimizer, "eps_abs" => ϵ, "eps_rel" => ϵ, "decompose" => true, "max_iter" => 50000))  # for larger graphs
    # model = Model(optimizer_with_attributes(SCS.Optimizer, "eps_abs" => ϵ, "eps_rel" => ϵ))
    # model = Model(optimizer_with_attributes(Mosek.Optimizer))
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
    println(eigmin(Matrix(Q_r)))
    return (X=X_r, Q=Q_r, value=val_r, lam=lam_r, Lam=Lam_r)
end

# Value funciton approximation
function valfun(Q)
    tol = 1e-6
    n = size(Q,1)-1
    i0 = n+1
    A = Symmetric(Q[1:n,1:n])
    b = Q[1:n,i0]
    return S -> b[S]' * pinv(A[S,S],rtol=tol) * b[S]  # pinv probably takes a significant portion of time. The pinv of sparse matrix is often dense. Can we do something else?
    # TODO: solve the SDP explicitly
end

del = (G,S,i) -> setdiff(S, vcat(neighbors(G,i), [i]))

# Rounding based on value function
function round_valfun(G, w, val, θ)
    n = length(w)
    S = collect(1:n)
    xr = falses(n)

    while length(S) > 0
        idx = argmax(w[j] + val(del(G,S,j)) for j in S)
        j = S[idx]
        xr[j] = 1
        S = del(G,S,j)
        println("Current weight: ", w'*xr)
    end
    return xr
end

function tabu_valfun(G, w, val, θ, ϵ=1e-6)
    n = length(w)
    S = collect(1:n)
    xr = falses(n)
    current_weight = 0

    # discard all nodes where v(i) > w_i
    T = copy(S)
    for v in T
        val_v = val([v])
        if w[v] < val_v - ϵ
            S = setdiff(S, v)
            println(v, " is discarded. Value: ", val_v)
        end
    end

    while !isempty(S)
        T = copy(S)
        for v in T
            val_v_c = val(del(G,T,v))
            if w[v] + val_v_c < θ - current_weight - ϵ
                S = setdiff(S, v)
                println(v, " is discarded. Value of LHS: ", w[v] + val_v_c, "; value of RHS: ", θ - current_weight)
            end
        end
        if !isempty(S)
            v = S[1]  # pick a vertex in the remaining candidates
            xr[v] = true
            S = del(G, S, v)
            current_weight = w' * xr
        end
        println("Current weight: ", current_weight)
    end
    return xr
end

#-----------------------------------------
graph_file = "hamming10-2.mtx"
A = mmread("../dat/" * graph_file)
G = complement(SimpleGraph(A))
n = nv(G)
println(n)
println(ne(G))

# Graph
E = collect(edges(G))

# Weight Vector
w = ones(n)

# Solve SDP
@time sol = dualSDP(E,w)
println("SDP Value: ", sol.value)
println("Eigvals: ", last(eigvals(sol.X), 3))

# Value fun & rounding
val = valfun(Matrix(sol.Q))  # convert SparseMatrix to Matrix

println("round_valfun begins")
xr = round_valfun(G, w, val, sol.value)

# verify
stable_set = findall(xr)
println(stable_set)
for i in stable_set
    for j in stable_set
        if has_edge(G, i, j)
            println("Failed! Edge ", i, " ", j)
        end
    end
end
println("Rounded Value: ", w' * xr)


println("tabu_valfun begins")
xt = tabu_valfun(G, w, val, sol.value, 1e-3)

stable_set = findall(xt)
println(stable_set)
for i in stable_set
    for j in stable_set
        if has_edge(G, i, j)
            println("Failed! Edge ", i, " ", j)
        end
    end
end
println("Rounded Value: ", w' * xt)
