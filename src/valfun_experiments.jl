using .ValFun

function get_params_valfun(G, w, params, use_qstab=false)
    if use_qstab
        return get_valfun_qstab(G, w, params[:use_all_cliques];
            solver=params[:solver],
            ϵ=params[:solver_ϵ],
            feas_ϵ=params[:solver_feas_ϵ]
        )
    else
        return get_valfun(G, w, params[:solve_dual], params[:formulation];
            solver=params[:solver],
            ϵ=params[:solver_ϵ],
            feas_ϵ=params[:solver_feas_ϵ],
            use_div=params[:use_div],
            pinv_rtol=params[:pinv_rtol]
        )
    end
end

function run_tabu_valfun(G, w, params, use_qstab=false)
    out = get_params_valfun(G, w, params, use_qstab)
    obj = out.sol.value
    println("Optimal value: ", obj)
    val = out.val
    @time x_stable, _ = tabu_valfun(G, w, obj, val; ϵ=params[:valfun_ϵ], verbose=params[:verbose])
    println("Retrieved value: ", w' * x_stable)
end

function run_tabu_valfun_test(G, w, params, use_qstab=false)
    out = get_params_valfun(G, w, params, use_qstab)
    obj = out.sol.value
    println("Optimal value: ", obj)
    val = out.val
    @time tabu_valfun_test(G, w, obj, val; ϵ=params[:valfun_ϵ], verbose=params[:verbose])
end

# Check whether the QSTAB LP interior point value function and the SDP value function result in the
# same operations when used by tabu_valfun_test()
function run_tabu_valfun_compare(G, w, sdp_params, qstab_params)
    sdp_out = get_params_valfun(G, w, sdp_params, false)
    θ = sdp_out.sol.value
    println("SDP value: ", θ)
    val_sdp = sdp_out.val

    qstab_out = get_params_valfun(G, w, qstab_params, true)
    qstab_sol = qstab_out.sol
    println("QSTAB LP value: ", qstab_sol.value)
    val_qstab = qstab_out.val

    tabu_valfun_compare(G, w, θ, val_sdp, val_qstab; ϵ=sdp_params[:valfun_ϵ], verbose=sdp_params[:verbose])
end

# QSTAB LP extreme points value functions verification
function run_test_qstab_valfuns(G, w, qstab_params, sdp_params)
    qstab_sol = qstab_lp_ext(G, w, qstab_params[:use_all_cliques];
        solver=qstab_params[:solver],
        ϵ=qstab_params[:solver_ϵ],
        feas_ϵ=qstab_params[:solver_feas_ϵ]
    )
    obj = qstab_sol.value
    println("QSTAB LP value: ", obj)
    # Run a tabu_valfun_test for each dual solutions in λ_ext_points, and count the number of failures
    failure_count = 0
    for (i, λ) in enumerate(qstab_sol.λ_ext_points)
        val_qstab = valfun_qstab(λ, qstab_sol.cliques)
        val_qstab_sdp = valfun(qstab_to_sdp(G, w, λ, qstab_sol.cliques); use_div=sdp_params[:use_div], pinv_rtol=sdp_params[:pinv_rtol])
        if !tabu_valfun_test(G, w, obj, val_qstab;
            ϵ=qstab_params[:valfun_ϵ],
            solver=qstab_params[:solver],
            solver_ϵ=qstab_params[:solver_ϵ],
            feas_ϵ=qstab_params[:solver_feas_ϵ]) || 
            !tabu_valfun_test(G, w, obj, val_qstab_sdp;
            ϵ=sdp_params[:valfun_ϵ],
            solver=sdp_params[:solver],
            solver_ϵ=sdp_params[:solver_ϵ],
            feas_ϵ=sdp_params[:solver_feas_ϵ])
            
            failure_count += 1
        end
    end
    println("Number of failed extreme points: ", failure_count, "; total extreme points: ", length(qstab_sol.λ_ext_points))
    λ_interior = sum(qstab_sol.λ_ext_points) / length(qstab_sol.λ_ext_points)
    val_qstab = valfun_qstab(λ_interior, qstab_sol.cliques)
    val_qstab_sdp = valfun(qstab_to_sdp(G, w, λ_interior, qstab_sol.cliques); use_div=sdp_params[:use_div], pinv_rtol=sdp_params[:pinv_rtol])
    println("Testing the interior point...")
    tabu_valfun_test(G, w, obj, val_qstab;
        ϵ=qstab_params[:valfun_ϵ],
        solver=qstab_params[:solver],
        solver_ϵ=qstab_params[:solver_ϵ],
        feas_ϵ=qstab_params[:solver_feas_ϵ],
        verbose=qstab_params[:verbose]
    )
    tabu_valfun_test(G, w, obj, val_qstab_sdp;
        ϵ=sdp_params[:valfun_ϵ],
        solver=sdp_params[:solver],
        solver_ϵ=sdp_params[:solver_ϵ],
        feas_ϵ=sdp_params[:solver_feas_ϵ],
        verbose=sdp_params[:verbose]
    )
end
