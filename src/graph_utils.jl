using Cairo, Fontconfig
using Graphs
using GraphPlot, Compose, Colors


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


function plot_graph_no_isolated(G, graph_name, use_complement)
    if use_complement
        image_file = "../images/" * graph_name * "_co.png"
    else
        image_file = "../images/" * graph_name * ".png"
    end

    V_no_isolated = vcat(filter(c -> length(c) > 1, connected_components(G))...)  # indices of vertices of G_S without isolated vertices
    println("Non-isolated vertices: ", length(V_no_isolated))
    if length(V_no_isolated) > 0
        G_no_isolated = G[V_no_isolated]
    else
        G_no_isolated = G
    end

    if is_bipartite(G_no_isolated)
        nodecolor = [colorant"lightseagreen", colorant"orange"]
        draw(PNG(image_file, 100cm, 100cm), gplot(G_no_isolated, NODESIZE=0.05/sqrt(nv(G_no_isolated)), layout=spring_layout, nodefillc=nodecolor[bipartite_map(G_no_isolated)]))
    else
        draw(PNG(image_file, 100cm, 100cm), gplot(G_no_isolated, NODESIZE=0.05/sqrt(nv(G_no_isolated)), layout=spring_layout))
    end
end
