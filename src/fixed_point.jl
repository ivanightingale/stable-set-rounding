using JuMP, SCS, COSMO, MosekTools

function get_interior_point(orig_model, orig_X, solver="SCS")
    model, reference_map = copy_model(orig_model)
    if solver == "SCS"
        set_optimizer(model, optimizer_with_attributes(SCS.Optimizer, "verbose" => false))
    elseif solver == "Mosek"
        set_optimizer(model, optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))  # for Benson johnson8-2-4, johnson8-4-4
    elseif solver == "COSMO"
        set_optimizer(model, optimizer_with_attributes(COSMO.Optimizer, "verbose" => true))
    end
    X = reference_map[orig_X]
    n = size(X)[1]
    @variable(model, t)
    @constraint(model, [t; 1; vec(X)] in MOI.LogDetConeSquare(size(X)[1]))
    @objective(model, Max, t)  # @objective(model, Max, logdet(X))
    optimize!(model)
    println(solution_summary(model))
    return value.(X)
end


function fixed_point_iteration(orig_model, orig_X, shift=nothing, ϵ=1e-6, solver="SCS")
    model, reference_map = copy_model(orig_model)
    if solver == "SCS"
        set_optimizer(model, optimizer_with_attributes(SCS.Optimizer, "verbose" => false, "eps_abs" => 1e-6, "eps_rel" => 1e-6))
    elseif solver == "COSMO"
        set_optimizer(model, optimizer_with_attributes(COSMO.Optimizer, "verbose" => true, "eps_abs" => 1e-6, "eps_rel" => 1e-6))
    end
    X = reference_map[orig_X]
    X_val = value.(orig_X)
    orig_obj = objective_function(model)
    if shift == nothing
        n = size(X)[1]
        shift = zeros((n,n))
    end

    terminate = false
    while !terminate
        prev_X = X_val
        @objective(model, Max, tr((prev_X + shift) * X))
        optimize!(model)
        X_val = value.(X)
        if norm(X_val - prev_X, Inf) < ϵ
            terminate = true
        end
        println(value(orig_obj))
    end
    println(value(orig_obj))
    return X_val
end
