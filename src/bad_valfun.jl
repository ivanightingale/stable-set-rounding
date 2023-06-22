using Combinatorics
using JuMP

using .ValFun
include("valfun_algorithms.jl")

# Solve an LP to find weights and a value function such that the value function satisfies some
# properties we know about the SDP and QSTAB LP value functions, and no vertex gets discarded
# by the value function at the current state. The value function is "bad" if, given the
# weights, there exists some vertex that is in fact not in any maximum stable set.
function find_bad_valfun(G; solver=:COPT, ϵ=0, feas_ϵ=0, verbose=false)
    n = nv(G)
    N = collect(1:n)

    model = Model()
    set_lp_optimizer(model, false; solver=solver, ϵ=ϵ, feas_ϵ=feas_ϵ, verbose=verbose)
    subsets = collect(powerset(N, 1))
    @variable(model, v[subsets] >= 0)
    @variable(model, w[1:n] >= 1)
    for s in powerset(N, 1)
        for i in setdiff(N, s)
            @constraint(model, v[s] <= v[sort(vcat(s, [i]))])  # Lemma 2(1)
        end
    end
    for s in powerset(N, 1, floor(Int64, n/2))
        disconnected_vertices = del_set(G, N, s)  # the set of vertices disconnected to s
        for t in powerset(setdiff(N, s), 1)
            if issubset(t, disconnected_vertices)
                @constraint(model, v[s] + v[t] == v[sort(vcat(s, t))])  # Lemma 2(2)
            end
        end
    end

    for i in N
        @constraint(model, v[N] - v[del(G, N, i)] == w[i])  # discard rule
        @constraint(model, v[[i]] >= w[i])  # Lemma 2(3)
    end
    # @constraint(model, v[Int64[]] == 0)
    @objective(model, Min, v[N])
    println("Optimizing...")
    optimize!(model)
    if termination_status(model) != MOI.OPTIMAL
        return nothing, nothing
    end

    bad_w = value.(w)
    bad_val = S -> isempty(S) ? 0 : value(v[S])

    return (w=bad_w, val=bad_val, obj=objective_value(model))
end

# function del_lp(G,S,i)
#     s_del = setdiff(S, vcat(neighbors(G,i), [i]))
#     if s_del == []
#         s_del = Int64[]
#     end
#     return s_del
# end

function del_set(G, S, vertices)
    s_del = setdiff(S, union([neighbors(G,i) for i in vertices]..., vertices))
    if s_del == []
        s_del = Int64[]
    end
    return s_del
end
