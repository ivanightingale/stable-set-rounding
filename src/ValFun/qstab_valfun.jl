export get_valfun_qstab, valfun_qstab, qstab_lp_ext

using PersistentCohomology  # for vietorisrips()
using Polyhedra
using Dualization

function get_valfun_qstab(G, w, use_all_cliques=true; solver=:COPT, ϵ=0, feas_ϵ=0, verbose=false)
    qstab_sol = qstab_lp_int(G, w, use_all_cliques; solver=solver, ϵ=ϵ, feas_ϵ=feas_ϵ, verbose=false)
    return (val=valfun_qstab(qstab_sol.λ, qstab_sol.cliques), sol=qstab_sol)
end

# Generate a value function from the optimal dual variables
function valfun_qstab(λ, cliques)
    return S -> sum([λ[i] for (i, c) in enumerate(cliques) if !isempty(S ∩ c)])
end

# Get the list of all maximal cliques (cliques that are not contained in larger cliques) of G
function maximal_cliques(G)
    n = nv(G)
    # find unweighted max stable set number
    α = max_clique(G, ones(n)).value
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
    return cliques
end

# Get the list of all cliques of G
function all_cliques(G)
    n = nv(G)
    # find unweighted max stable set number
    α = max_clique(G, ones(n)).value
    cliques = Vector{Vector{Int64}}(undef, 0)
    # obtain a tuple where the i-th entry contains all cliques of G with size i
    clique_lists_by_size = vietorisrips(adjacency_matrix(G), α)

    for k in 1:α
        k_cliques = keys(clique_lists_by_size[k])
        for c in k_cliques
            clique_to_add = sort(collect(c))
            push!(cliques, clique_to_add)
        end
    end
    return cliques
end

# Solve max stable set by solving an LP over the clique polytope (QSTAB), constructed by adding
# constraints for either all cliques or all maximal cliques.
# Return all dual optimal BFS and the corresponding cliques in the same order.
function qstab_lp_ext(G, w, use_all_cliques=true; solver=:COPT, ϵ=0, feas_ϵ=0, verbose=false)
    n = nv(G)
    E = collect(edges(G))
    model = Model()
    set_lp_optimizer(model, false; solver=solver, ϵ=ϵ, feas_ϵ=feas_ϵ, verbose=verbose)
    @variable(model, x[1:n] >= 0)

    cons = Vector{ConstraintRef}(undef, 0)
    if use_all_cliques
        cliques = all_cliques(G)
    else
        cliques = maximal_cliques(G)
    end
    # add corresponding constraints
    for c in cliques
        push!(cons, @constraint(model, sum(x[c]) <= 1))
    end
    @objective(model, Max, w' * x)
    optimize!(model)

    λ_ext_points = dual_extreme_points(model)

    if verbose
        println("Dual extreme points:")
        for λ in λ_ext_points
            println(cliques[findall(λ .> 0.5)])
        end
    end

    return (x=value.(x), value=objective_value(model), λ_ext_points=λ_ext_points, cliques=cliques)
end

# Solve max stable set by solving an LP over the clique polytope (QSTAB), constructed by adding
# constraints for either all cliques or all maximal cliques.
# Return an interior point dual solution and the corresponding cliques in the same order.
function qstab_lp_int(G, w, use_all_cliques=true; solver=:Mosek, ϵ=0, feas_ϵ=0, verbose=false)
    n = nv(G)
    E = collect(edges(G))
    model = Model()
    @variable(model, x[1:n] >= 0)

    cons = Vector{ConstraintRef}(undef, 0)
    if use_all_cliques
        cliques = all_cliques(G)
    else
        cliques = maximal_cliques(G)
    end

    # add corresponding constraints
    for c in cliques
        push!(cons, @constraint(model, sum(x[c]) <= 1))
    end
    @objective(model, Max, w' * x)
    dual_sol = dual_interior_point(model; solver=solver, ϵ=ϵ, feas_ϵ=feas_ϵ, verbose=verbose)

    return (value=dual_sol.value, λ=dual_sol.λ, cliques=cliques)
end

# Solve an IP to find a max clique on G given weight w
# Return the optimal solution
function max_clique(G, w; solver=:COPT, ϵ=0, feas_ϵ=0, verbose=false)
    n = nv(G)
    model = Model()
    set_lp_optimizer(model, false; solver=solver, ϵ=ϵ, feas_ϵ=feas_ϵ, verbose=verbose)
    @variable(model, z[1:n], Bin)
    @constraint(model, [e in edges(complement(G))], z[src(e)] + z[dst(e)] <= 1)
    @objective(model, Max, w' * z)
    optimize!(model)
    return (z=value.(z), value=Int(objective_value(model)))
end

# Find all extreme points of the optimal face in the dual of model. The model needs to be
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
    return -opt_ext_points
end

# Solve the dual of a model using interior point method to obtain a point in the relative interior
# of thedual optimal face
function dual_interior_point(model; solver=:Mosek, ϵ=0, feas_ϵ=0, verbose=false)
    println("Dualizing...")
    dual_model = dualize(model)
    set_lp_optimizer(dual_model, true; solver=solver, ϵ=ϵ, feas_ϵ=feas_ϵ, verbose=verbose)
    optimize!(dual_model)
    return (λ=-value.(all_variables(dual_model)), value=objective_value(dual_model))
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

# Solve max stable set by starting with the fractional stable set polytope (edge
# polytope) and adding clique constraint cutting planes by solving auxiliary problems.
# Return all optimal dual BFS and the corresponding cliques.
function qstab_lp_cutting_planes(G, w; solver=:SCS, ϵ=0, feas_ϵ=0, verbose=false)
    n = nv(G)
    E = collect(edges(G))
    model = Model()
    set_lp_optimizer(model; solver=solver, ϵ=ϵ, feas_ϵ=feas_ϵ, verbose=verbose)
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
