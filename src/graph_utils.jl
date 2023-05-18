using GraphPlot, Compose, Colors
using Cairo, Fontconfig
using Graphs
using MatrixMarket, DelimitedFiles
using Random
using Combinatorics


# chordal, petersen
# graphs generated as adjacency matrix text
function load_family_graph(graph_name, family, use_complement=false)
    A = readdlm("../dat/" * family * "/" * graph_name * ".txt")
    if use_complement
        G = complement(SimpleGraph(A))  # the complement of chordal/perfect graph is also perfect
    else
        G = SimpleGraph(A)
    end
    return G
end


function load_dimacs_graph(graph_name, use_complement=true)
    A = mmread("../dat/dimacs/" * graph_name * ".mtx")
    if use_complement
        G = complement(SimpleGraph(A))  # the original DIMACS graphs are test cases for max clique problem, so use the complement
    else
        G = SimpleGraph(A)
    end
    return G
end

function generate_family_graph(family, n, use_complement=false; k=n)
    if family == "wheel"
        graph_name = "wheel-" * string(n)
        G = wheel_graph(n)
    elseif family == "hole"
        graph_name = "hole-" * string(n)
        G = cycle_graph(n)
    elseif family == "path"
        graph_name = "path-" * string(n)
        G = path_graph(n)
    elseif family == "chain"
        graph_name = string(k) * "-chain-" * string(n)
        G = SimpleGraph(n, 0)
        for i in 1 : (n / 2)
            for j in 1:k
                add_edge!(G, (i, n / 2 + 1 + mod(i + j - 2, n / 2)))
            end
        end
    end
    if use_complement
        G = complement(G)
    end
    return G, graph_name
end


function plot_graph(G, graph_name, use_complement=false; S_to_color=[], remove_isolated=false, suffix="", layout = spring_layout, add_label=false)
    if suffix != ""
        suffix = "_" * suffix
    end
    if use_complement
        suffix = "_co" * suffix
    end

    image_file = "../images/" * graph_name * suffix * ".png"

    if remove_isolated
        G_copy = copy(G)
        # indices of vertices of G without isolated vertices
        V_no_isolated = vcat(filter(c -> length(c) > 1, connected_components(G))...)
        if length(V_no_isolated) > 0  # if all vertices are isolated, keep them instead
            G_copy = G[V_no_isolated]
        end
    else
        G_copy = G
    end

    nodecolor = [colorant"lightseagreen", colorant"orange"]
    if add_label
        node_label = 1:nv(G_copy)
    else
        node_label = nothing
    end

    # FIXME: why sometimes doesn't save when run from REPL?

    if length(S_to_color) == 0
        if is_bipartite(G_copy)
            draw( PNG(image_file, 100cm, 100cm), gplot(G_copy, NODESIZE=0.05/sqrt(nv(G_copy)), layout=layout, nodefillc=nodecolor[bipartite_map(G_copy)], nodelabel=node_label) )
        else
            draw( PNG(image_file, 100cm, 100cm), gplot(G_copy, NODESIZE=0.05/sqrt(nv(G_copy)), layout=layout, nodelabel=node_label) )
        end
    else
        # plot the specified set of vertices S_to_color in a different color
        V = vertices(G_copy)
        color_map = [V[i] in S_to_color for i in 1:nv(G_copy)] .+ 1
        draw( PNG(image_file, 100cm, 100cm), gplot(G_copy, NODESIZE=0.05/sqrt(nv(G_copy)), layout=layout, nodefillc=nodecolor[color_map], nodelabel=node_label) )
    end
end

# TODO
# function parse_psm()
# savegraph
# end

function generate_generalized_split_graph(n)
    b = rand([1,-1])
    E = edges(erdos_renyi(n, 0.5))
    k = rand(0:n-1)  # first k vertices are put in the central clique C; exclude the case where C includes all n vertices
    # partition the remaining (n-k) vertices into side cliques S
    n_partitions = rand(1:n-k)
    partitions_assignment = rand(1:n_partitions, n-k)
    partitions = [k .+ findall(partitions_assignment .== i) for i = 1:n_partitions]

    G = SimpleGraph(n)
    # central clique C
    for e in combinations(1:k, 2)
        add_edge!(G, e...)
    end
    # side cliques S
    println(n_partitions)
    println(partitions)
    for i in 1:n_partitions
        for e in combinations(partitions[i], 2)
            add_edge!(G, e...)
        end
    end

    # add edges in E that are between C and S
    for e in E
        if src(e) <= k
            add_edge!(G, e)
        end
    end

    if b < 0
        return complement(G)
    end
    plot_graph(G, "test_gsg"; add_label=true)
    return G
end