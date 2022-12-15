using GraphPlot, Compose, Colors
using Cairo, Fontconfig
using Graphs
using MatrixMarket, DelimitedFiles


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
    A = mmread("../dat/" * graph_name * ".mtx")
    if use_complement
        G = complement(SimpleGraph(A))  # the original DIMACS graphs are test cases for max clique problem, so use the complement
    else
        G = SimpleGraph(A)
    end
    return G
end

function generate_family_graph(family, n, use_complement=false)
    if family == "wheel"
        graph_name = "wheel-" * string(n)
        G = wheel_graph(n)
    elseif family == "hole"
        graph_name = "hole-" * string(n)
        G = cycle_graph(n)
    end
    if use_complement
        G = complement(G)
    end
    return G, graph_name
end


# TODO: allow to input a subset of vertices to put in a different color (if not provided, check bipartite)
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
        # println("Non-isolated vertices: ", length(V_no_isolated))
        if length(V_no_isolated) > 0
            G_copy = G[V_no_isolated]
        else
            G_copy = G
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
        V = vertices(G_copy)
        color_map = [V[i] in S_to_color for i in 1:nv(G_copy)] .+ 1
        draw( PNG(image_file, 100cm, 100cm), gplot(G_copy, NODESIZE=0.05/sqrt(nv(G_copy)), layout=layout, nodefillc=nodecolor[color_map], nodelabel=node_label) )
    end
end
