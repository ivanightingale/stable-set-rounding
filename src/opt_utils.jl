using Graphs
using StatsBase
using JuMP, SCS, COSMO, MosekTools

function theta_sdp_model(solver="SCS")
    if solver == "SCS"
	       # return Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => false, "eps_abs" => 1e-6, "eps_rel" => 1e-6))
           return Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => false, "eps_rel" => 1e-7, "eps_abs" => 1e-7))
    elseif solver == "COSMO"
        return Model(optimizer_with_attributes(COSMO.Optimizer, "verbose" => true, "eps_abs" => 1e-6, "eps_rel" => 1e-6, "decompose"=> true))  # TODO: why no decomposition?
    elseif solver == "Mosek"
        return Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))  # TODO: why no decomposition?
    end
end

function hyperplane_rounding(X, G, n_iter=5000)
    max_val = -Inf
    max_x = nothing
    i0 = size(X)[1]
    X_sym = Symmetric(X)
    cho = cholesky(sparse(X_sym), shift = -eigmin(X_sym) + 1e-9)  # FIXME: may fail when rank is deficient. A workaround: https://discourse.julialang.org/t/cholesky-decomposition-of-low-rank-positive-semidefinite-matrix/70397/3; however, the method mentioned in the post usually terminates early and results in incorrect resuult.
    V_permutation = invperm(cho.p)
    V = sparse(cho.L)[V_permutation, V_permutation]  # X = V V'
    for i = 1:n_iter
        u = normalize(randn(i0))
        v = V' * u
        x = sign.(v)

        for e in edges(G)
        	i = src(e)
        	j = dst(e)
        	if abs(x[i] + x[j] + x[i0]) != 1
        		if abs(v[i] - v[i0]) > abs(v[j] - v[i0])
        			x[i] = -x[i]
        		else
        			x[j] = -x[j]
        		end
        	end
        end

        # check error
		for e in edges(G)
			i = src(e)
			j = dst(e)
			if abs(x[i] + x[j] + x[i0]) != 1
				println("hi")
			end
		end

        current_val = benson_cost(x)
        if current_val > max_val
            max_val = current_val
            max_x = x
        end
    end

    return max_x, max_val
end

function benson_cost(x)
    i0 = size(x)[1]
    count(i -> i == x[i0], x) - 1
end

function greedy_stable_set_rounding(X, G, n_iter=5000)
    max_val = -Inf
    max_x = nothing
    n = nv(G)
    for i = 1:n_iter
        V = collect(1:n)
        x = zeros(n)
        p = diag(X)
        while !isempty(V)
            current_v = sample(V, ProbabilityWeights(p))
            x[current_v] = 1
            v_filter = findall(v -> v ∉ current_v ∪ neighbors(G, current_v), V)
            V = V[v_filter]
            p = p[v_filter]
        end

        current_val = sum(x)
        if current_val > max_val
            max_val = current_val
            max_x = x
        end
    end
    return max_x, max_val
end