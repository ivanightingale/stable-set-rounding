using JuMP, SCS, COSMO, MosekTools
using LinearAlgebra, SparseArrays
using Combinatorics
using Graphs
using GraphPlot, Compose
import Cairo, Fontconfig
using Colors

# Solve dual SDP
function dualSDP(E,w)
    n = length(w)
    i0 = n + 1
    ϵ = 1e-7
    # model = Model(optimizer_with_attributes(COSMO.Optimizer, "eps_abs" => ϵ, "eps_rel" => ϵ, "decompose" => true, "max_iter" => 1000000))  # for larger graphs
    model = Model(optimizer_with_attributes(SCS.Optimizer, "eps_abs" => ϵ, "eps_rel" => ϵ, "max_iters" => 1000000, "verbose" => 0))
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
        println("Current weight: ", w' * xr)
    end
    return xr
end


# iteratively apply discard rules and pick the first candidate
function tabu_valfun(G, w, val, θ, ϵ=1e-6)
    n = length(w)
    S = collect(1:n)
    xr = falses(n)
    current_weight = 0

    S = discard_rule_1(S, val, w, ϵ)
    S = discard_rule_2(S, G, θ, val, w, ϵ)

    while !isempty(S)
        S = discard_rule_2(S, G, θ, val, w, ϵ, current_weight)

        if !isempty(S)
            v = S[1]  # pick a vertex in the remaining candidates
            xr[v] = true
            S = del(G, S, v)  # this ensures the final output is a stable set
            current_weight = w' * xr
        end
        println("Current weight: ", current_weight)
    end
    return xr
end


function plot_graph_no_isolated(G, graph_name, use_complement)
    if use_complement
        image_file = "../images/" * graph_name * "_co.png"
    else
        image_file = "../images/" * graph_name * ".png"
    end

    V_no_isolated = vcat(filter(c -> length(c) > 1, connected_components(G))...)  # indices of vertices of G_S without isolated vertices
    println("Non-isolated vertices: ", length(V_no_isolated))
    if length(V_no_isolated) > 0
        G_no_isolated = G[V_no_isolated]
    else
        G_no_isolated = G
    end
    if is_bipartite(G_no_isolated)
        nodecolor = [colorant"lightseagreen", colorant"orange"]
        draw(PNG(image_file, 100cm, 100cm), gplot(G_no_isolated, NODESIZE=0.05/sqrt(nv(G_no_isolated)), layout=spring_layout, nodefillc=nodecolor[bipartite_map(G_no_isolated)]))
    else
        draw(PNG(image_file, 100cm, 100cm), gplot(G_no_isolated, NODESIZE=0.05/sqrt(nv(G_no_isolated)), layout=spring_layout))
    end
end


function tabu_valfun_verify_I(G, w, val, θ, ϵ, graph_name, use_complement)
    n = length(w)
    S = collect(1:n)
    xr = falses(n)
    current_weight = 0

    S = discard_rule_1(S, val, w, ϵ)
    S = discard_rule_2(S, G, θ, val, w, ϵ)

    println("First round complete. Remaining size: ", length(S))

    plot_graph_no_isolated(G[S], graph_name, use_complement)

    T = copy(S)

    for first_v in T
        xr[first_v] = true
        S = del(G, S, first_v)
        current_weight = w' * xr
        while !isempty(S)
            S = discard_rule_2(S, G, θ, val, w, ϵ, current_weight, false)

            if !isempty(S)
                v = S[1]  # pick a vertex in the remaining candidates
                xr[v] = true
                S = del(G, S, v)  # this ensures the final output is a stable set
                current_weight = w' * xr
            end
        end
        println("Finished starting with ", string(first_v), ". Final weight: ", w' * xr)
        xr = falses(n)
        S = copy(T)
        current_weight = 0
    end

    for first_v in T
        xr[first_v] = true
        S = del(G, S, first_v)
        E = collect(edges(G[S]))
        sol = dualSDP(E, w[S])
        val_S = val(S)
        println("Original theta: ", θ, " Value function when ", string(first_v), " is picked: ", val_S, "; new theta value: ", sol.value)
        if abs(val_S - sol.value) > ϵ || abs(θ - sol.value) > 1 + ϵ
            println("Warning!")
        end
        S = copy(T)
    end
end


function discard_rule_1(S, val, w, ϵ, verbose=true)
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


function discard_rule_2(S, G, θ, val, w, ϵ, current_weight=0, verbose=true)
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


function load_dimacs_graph(graph_name, use_complement=true)

    A = mmread("../dat/" * graph_name * ".mtx")
    if use_complement
        G = complement(SimpleGraph(A))  # the original DIMACS graphs are test cases for max clique problem, so use the complement
    else
        G = SimpleGraph(A)
    end
    return G
end

function load_chordal_graph(graph_name, use_complement=false)

    A = readdlm("../dat/chordal/" * graph_name * ".txt")
    if use_complement
        G = complement(SimpleGraph(A))  # the complement of chordal/perfect graph is also perfect
    else
        G = SimpleGraph(A)
    end
    return G
end


#-----------------------------------------
using MatrixMarket
use_complement = false
graph_name = "hamming8-2"
G = load_dimacs_graph(graph_name, use_complement)

# using DelimitedFiles
# use_complement = true
# graph_name = "connecting-100-2"
# G = load_chordal_graph(graph_name, use_complement)


n = nv(G)
println(n)
println(ne(G))
E = collect(edges(G))

# Weight Vector
w = ones(n)

# Solve SDP
@time sol = dualSDP(E, w)
println("SDP Value: ", sol.value)
# println("Eigvals: ", last(eigvals(sol.X), 3))

# Value fun & rounding
val = valfun(Matrix(sol.Q))  # convert SparseMatrix to Matrix

# println("round_valfun begins")
# xr = round_valfun(G, w, val, sol.value)
# println(findall(xr))
# println("Rounded Value: ", w' * xr)


# println("tabu_valfun begins")
# xt = tabu_valfun(G, w, val, sol.value, 1e-3)
# println(findall(xt))
# println("Rounded Value: ", w' * xt)

tabu_valfun_verify_I(G, w, val, sol.value, 1e-3, graph_name, use_complement)
