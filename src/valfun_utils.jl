using JuMP, MosekTools, SCS, COSMO, COPT
using LinearAlgebra, SparseArrays
using Combinatorics
using Graphs

function theta_sdp_model(;solver="SCS", ϵ=0, feas_ϵ=0, verbose=false)
    if solver == "COSMO"  # for larger graphs
        model = Model(optimizer_with_attributes(COSMO.Optimizer, "decompose" => true, "max_iter" => 1000000, "verbose" => verbose))
        if ϵ > 0
            set_optimizer_attribute(model, "eps_abs", ϵ)
            set_optimizer_attribute(model, "eps_rel", ϵ)
        end
        if feas_ϵ > 0
            set_optimizer_attribute(model, "eps_prim_inf", feas_ϵ)
            set_optimizer_attribute(model, "eps_dual_inf", feas_ϵ)
        end
    elseif solver == "SCS"
        model = Model(optimizer_with_attributes(SCS.Optimizer, "max_iters" => 1000000, "verbose" => verbose))
        if ϵ > 0
            set_optimizer_attribute(model, "eps_abs", ϵ)
            set_optimizer_attribute(model, "eps_rel", ϵ)
        end
        if feas_ϵ > 0
            set_optimizer_attribute(model, "eps_infeas", feas_ϵ)
        end
    elseif solver == "Mosek"
        model = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => !verbose))
        if ϵ > 0
            set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", ϵ)
            set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_MU_RED", ϵ)
        end
        if feas_ϵ > 0
            set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", feas_ϵ)  # TODO: what are the differences?
            set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", feas_ϵ)
            set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_INFEAS", feas_ϵ)
        end
    elseif solver == "COPT"
        model = Model(optimizer_with_attributes(COPT.ConeOptimizer, "Logging" => verbose, "LogToConsole" => false))
        if ϵ > 0
            set_optimizer_attribute(model, "AbsGap", ϵ)
            set_optimizer_attribute(model, "RelGap", ϵ)
        end
        if feas_ϵ > 0
            set_optimizer_attribute(model, "FeasTol", feas_ϵ)
        end
    end
    return model
end

function qstab_lp_model(;solver="SCS", ϵ=0, feas_ϵ=0, verbose=false)
    if solver in ["COSMO", "SCS"]
        model = theta_sdp_model(;solver=solver, ϵ=ϵ, feas_ϵ=feas_ϵ, verbose=verbose)
    elseif solver == "Mosek"
        model = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => !verbose))
        if ϵ > 0
            set_optimizer_attribute(model, "MSK_DPAR_INTPNT_TOL_REL_GAP", ϵ)
            set_optimizer_attribute(model, "MSK_DPAR_INTPNT_TOL_MU_RED", ϵ)
        end
        if feas_ϵ > 0
            set_optimizer_attribute(model, "MSK_DPAR_INTPNT_TOL_PFEAS", feas_ϵ)  # TODO: what are the differences?
            set_optimizer_attribute(model, "MSK_DPAR_INTPNT_TOL_DFEAS", feas_ϵ)
            set_optimizer_attribute(model, "MSK_DPAR_INTPNT_TOL_INFEAS", feas_ϵ)
        end
    elseif solver == "COPT"
        model = Model(optimizer_with_attributes(COPT.Optimizer, "Logging" => verbose, "LogToConsole" => false))
        if ϵ > 0
            set_optimizer_attribute(model, "AbsGap", ϵ)
            set_optimizer_attribute(model, "RelGap", ϵ)
        end
        if feas_ϵ > 0
            set_optimizer_attribute(model, "FeasTol", feas_ϵ)
        end
    end
    return model
end


function print_valfun(val, n, max_size=n)
    subsets = powerset(1:n, 0, max_size)
    for s in subsets
        println(s, " ", val(s))
    end
end


del = (G,S,i) -> setdiff(S, vcat(neighbors(G,i), [i]))
del! = (G,S,i) -> setdiff!(S, vcat(neighbors(G,i), [i]))


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


function sdp_to_qstab(Q, w, cliques; solver="SCS", ϵ=0, feas_ϵ=0, verbose=false)
    n = length(w)
    i0 = n + 1
    model = qstab_lp_model(solver=solver, ϵ=ϵ, feas_ϵ=feas_ϵ, verbose=verbose)
    @variable(model, λ[cliques] >= 0)
    @constraint(model, sum(λ) == Q[i0, i0])
    @constraint(model, [i in 1:n], 2 * sum(λ[c] for c in cliques if i in c) == Q[i, i] + w[i])
    for i in 1:n
        for j in 1:(i - 1)
            @constraint(model, sum(λ[c] for c in cliques if i in c && j in c) == Q[i, j])
        end
    end
    # display(all_constraints(model; include_variable_in_set_constraints = true))
    # println()
    optimize!(model)
    println(solution_summary(model))
    return value.(λ)
end