using JuMP, MosekTools, SCS, COSMO, COPT, Dualization
using LinearAlgebra, SparseArrays
using Combinatorics
using Graphs
using Polyhedra
using PersistentCohomology

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
function valfun_sdp_explicit(Q; solver="SCS", ϵ=1e-7)
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

# Solve max stable set by solving an LP over the clique polytope (QSTAB) by
# finding all maximal stable sets (stable sets that are not subsets of other stable
# sets) and adding corresponding constraints.
# Return all optimal dual BFS and the corresponding cliques
function qstab_lp(G, w; verbose=false)
    n = nv(G)
    E = collect(edges(G))
    model = Model(optimizer_with_attributes(COPT.Optimizer, "Logging" => verbose, "LogToConsole" => verbose))
    @variable(model, x[1:n] >= 0)

    # find unweighted max stable set number
    α = Int(max_clique(G, ones(n)).value)

    cons = Vector{ConstraintRef}(undef, 0)
    cliques = Vector{Vector{Int64}}(undef, 0)

    # obtain a tuple where the i-th entry contains all cliques of G with size i
    clique_lists_by_size = vietorisrips(adjacency_matrix(G), α)
    # find all maximal cliques
    for k in α:-1:1
        k_cliques = keys(clique_lists_by_size[k])
        for c in k_cliques
            clique_to_add = sort(collect(c))
            to_add = true
            for existing_clique in cliques
                # do not add the new clique if it is a subset of an already added clique
                if issubset(clique_to_add, existing_clique)
                    to_add = false
                    break
                end
            end
            if to_add
                push!(cliques, clique_to_add)
            end
        end
    end
    # add corresponding constraints
    for c in cliques
        push!(cons, @constraint(model, sum(x[c]) <= 1))
    end
    println(cliques)
    @objective(model, Max, w' * x)
    optimize!(model)

    return (x=value.(x), value=objective_value(model), λ_ext_points=dual_extreme_points(model), cliques=cliques)
end

# Solve max stable set by starting with the fractional stable set polytope (edge
# polytope) and adding clique constraint cutting planes by solving auxiliary problems.
# Return all optimal dual BFS and the corresponding cliques.
function qstab_lp_cutting_planes(G, w; verbose=false)
    n = nv(G)
    E = collect(edges(G))
    model = Model(optimizer_with_attributes(COPT.Optimizer, "Logging" => verbose, "LogToConsole" => verbose))
    @variable(model, x[1:n] >= 0)

    cons = Vector{ConstraintRef}(undef, 0)
    cliques = Vector{Vector{Int64}}(undef, 0)
    for e in edges(G)
        push!(cons, @constraint(model, x[src(e)] + x[dst(e)] <= 1))
        push!(cliques, [src(e), dst(e)])
    end
    @objective(model, Max, w' * x)
    sub_sol_value = Inf
    while sub_sol_value > 1
        optimize!(model)
        # solve auxiliary max clique problem
        sub_sol = max_clique(G, value.(x))
        sub_sol_value = sub_sol.value
        if sub_sol_value > 1
            clique_to_add = findall(sub_sol.z .> 0.5)
            println("Clique to add:")
            println(clique_to_add)
            # remove redundant constraints of cliques that are subsets of the new clique
            index_to_delete = findall(map((c) -> issubset(c, clique_to_add), cliques))
            println("Deleting...")
            for i in sort(index_to_delete, rev=true)
                println(cliques[i])
                delete(model, cons[i])
                deleteat!(cons, i)
                deleteat!(cliques, i)
            end
            push!(cliques, clique_to_add)
            push!(cons, @constraint(model, sum(x[clique_to_add]) <= 1))
        end
    end
    return (x=value.(x), value=objective_value(model), λ_ext_points=dual_extreme_points(model), cliques=cliques)
end


# Solve an IP to find a max clique on G given weight w
# Return the optimal solution
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


# Find all extreme points of the optimal face in the dual of model. Model needs to be
# already solved to optimality.
function dual_extreme_points(model)
    opt_val = objective_value(model)
    println("Dualizing...")
    dual_prob = dualize(model)
    # optimality face
    @constraint(dual_prob, sum(all_variables(dual_prob)) == -opt_val)  # note the negative sign
    dual_opt_h = hrep(dual_prob)
    println("H to V...")
    dual_opt_v = doubledescription(dual_opt_h)
    println("Computing extreme points...")
    opt_ext_points = collect(Polyhedra.points(dual_opt_v))
    # println(opt_ext_points)
    return -opt_ext_points
end


# convert an optimal LP solution to max stable set to an optimal solution of the SDP relaxation
function qstab_to_sdp(G, w, λ, cliques)
    n = nv(G)
    i0 = n + 1
    E = collect(edges(G))
    val = valfun_qstab(λ, cliques)
    v = [val(i) for i in 1:n]
    Q = sparse(collect(1:n), fill(i0, n), -v, i0, i0) + 2 * sparse(collect(1:n), collect(1:n), v, i0, i0) - Diagonal([w; 0])
    for e in E
        i = src(e)
        j = dst(e)
        Q[i, j] = sum([λ[k] for (k, c) in enumerate(cliques) if(i in c && j in c)])
    end
    Q[i0, i0] = sum(λ)
    Q = Symmetric(Q)
    # display(Q)
    return Matrix(Q)
end


function sdp_to_qstab(Q, w, cliques; solver="SCS")
    n = length(w)
    i0 = n + 1
    model = theta_sdp_model(solver=solver)
    @variable(model, λ[cliques] >= 0)
    # @constraint(model, [i in 1:length(w)], sum(λ[c] for c in cliques if i in c) >= w[i])
    @constraint(model, sum(λ) == Q[i0, i0])
    @constraint(model, [i in 1:length(w)], 2 * sum(λ[c] for c in cliques if i in c) == Q[i, i] + w[i])
    for i in 1:length(w)
        for j in 1:(i - 1)
            @constraint(model, sum(λ[c] for c in cliques if i in c && j in c) == Q[i, j])
        end
    end
    optimize!(model)
    # println(value.(λ))
    println(solution_summary(model))
end