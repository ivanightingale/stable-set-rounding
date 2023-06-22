using GraphPlot, Compose, Colors, Cairo, Fontconfig
using Graphs, GraphIO
using MatrixMarket, DelimitedFiles
using Random
using Combinatorics

function load_dimacs_graph(graph_name, use_complement=true)
    A = mmread("dat/dimacs/" * graph_name * ".mtx")
    if use_complement
        G = complement(SimpleGraph(A))  # the original DIMACS graphs are test cases for max clique problem, so use the complement
    else
        G = SimpleGraph(A)
    end
    return G
end

# Load a graph from its adjacency matrix file in the dat/family folder
# chordal: chordal graphs generated by RandomChordalGraph() in Sage
# perfect: currently contains subgraphs of above chordal graphs, or modified versions of such,
# that are found to be interesting in some ways (when used with pre-specified weights in 
# run_experiment.jl, which were generated by find_bad_valfun()) 
# petersen: Petersen graphs generated by GeneralizedPetersenGraph() in Sage
# random: small random graphs generated by RandomGNP() in Sage. "np": not weakly perfect; "qp": weakly perfect but not 
# perfect (we call it "quasi-perfect"); "p": perfect.
function load_family_graph(graph_name, family, use_complement=false)
    A = readdlm("dat/" * family * "/" * graph_name * ".txt")
    if use_complement
        G = complement(SimpleGraph(A))  # the complement of chordal/perfect graph is also perfect
    else
        G = SimpleGraph(A)
    end
    return G
end

# Generate a graph available in JuliaGraphs
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
        # Get indices of vertices of G which are not isolated
        V_no_isolated = vcat(filter(c -> length(c) > 1, connected_components(G))...)
        if length(V_no_isolated) > 0  # If all vertices are isolated, keep them instead
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

    if length(S_to_color) == 0
        if is_bipartite(G_copy)
            draw( PNG(image_file, 100cm, 100cm), gplot(G_copy, NODESIZE=0.05/sqrt(nv(G_copy)), layout=layout, nodefillc=nodecolor[bipartite_map(G_copy)], nodelabel=node_label) )
        else
            draw( PNG(image_file, 100cm, 100cm), gplot(G_copy, NODESIZE=0.05/sqrt(nv(G_copy)), layout=layout, nodelabel=node_label) )
        end
    else
        # Plot the specified set of vertices S_to_color in a different color
        V = vertices(G_copy)
        color_map = [V[i] in S_to_color for i in 1:nv(G_copy)] .+ 1
        draw( PNG(image_file, 100cm, 100cm), gplot(G_copy, NODESIZE=0.05/sqrt(nv(G_copy)), layout=layout, nodefillc=nodecolor[color_map], nodelabel=node_label) )
    end
end

# TODO: function to parse miplib files
# function parse_mps()
# end

# Generate a generalized split graph.
function generate_generalized_split_graph(n, file_name)
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

    # Add edges in E that are between C and S
    for e in E
        if src(e) <= k
            add_edge!(G, e)
        end
    end

    if b < 0
        return complement(G)
    end
    # plot_graph(G, "test_gsg"; add_label=true)
    savegraph("dat/gsg/" * file_name, G)
    return G
end

function load_gsg_graph(graph_name, use_complement=false)
    G = loadgraph("dat/gsg/" * graph_name)
    if use_complement
        G = complement(G)
    end
    return G
end
