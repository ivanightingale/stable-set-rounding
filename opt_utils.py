import numpy as np
import cvxpy as cp
import networkx as nx
from linalg_utils import normalize_rows, eigh_proj, decompose_psd


# project each entry of vector v into the annulus with inner radius min_r and outer radius max_r centered at 0
def proj_to_annulus(v, min_r, max_r):
    # find the closest point of each entry in the ring
    n = v.shape[0]
    assert np.isscalar(min_r) or len(min_r) == n
    assert np.isscalar(max_r) or len(max_r) == n
    x = np.zeros(n)
    norms = np.linalg.norm(v, axis=1)

    for i in range(n):
        v_i = v[i]
        x_i = None
        min_r_i = min_r if np.isscalar(min_r) else min_r[i]
        max_r_i = max_r if np.isscalar(max_r) else max_r[i]

        if norms[i] < min_r_i:
            x_i = v_i / norms[i] * min_r_i
        elif norms[i] > max_r_i:
            x_i = v_i / norms[i] * max_r_i
        else:
            x_i = v_i
        x[i] = [x_i]

    return x


# round each row of Y to the unit annulus, or project onto the annulus between min_r and max_r
def hyperplane_rounding(Y, cost, min_r=1, max_r=1, n_iter=100, is_complex=False):
    min_cost = np.Inf
    best_x = None
    d = Y.shape[1]
    rng = np.random.default_rng()
    for i in range(n_iter):
        if is_complex:
            r = 1 / np.sqrt(2) * rng.standard_normal((d, 1)) + 1 / np.sqrt(2) * rng.standard_normal((d, 1)) * 1j
        else:
            r = rng.standard_normal((d, 1))
        if np.isscalar(min_r) and np.isscalar(max_r):
            if min_r == 1 and max_r == 1:
                if is_complex:
                    x = normalize_rows(Y @ r)
                else:
                    x = np.sign(Y @ r)
            else:
                x = proj_to_annulus(Y @ r, min_r, max_r)
        else:
            x = proj_to_annulus(Y @ r, min_r, max_r)

        cost_val = cost(x)
        if cost_val < min_cost:
            min_cost = cost_val
            best_x = x
    return min_cost, best_x


# approximately reduce rank of X = YY* by delta_rank via eigenprojection and row normalization
# if can't be reduced anymore, return the original Y
def elliptope_eigen_proj(Y, delta_rank=1):
    assert len(Y) > delta_rank
    X = np.outer(Y, Y.conj())
    current_rank = np.linalg.matrix_rank(X, tol=1e-9)
    target_rank = current_rank - delta_rank
    Y_proj = Y
    new_rank = current_rank
    if target_rank >= 1:
        new_rank = target_rank
        eigen_val, eigen_vec = eigh_proj(X, delta_rank)

        Y_proj = eigen_vec[:, range(new_rank)] @ np.diag(np.sqrt(eigen_val[range(new_rank)]))
        Y_proj = normalize_rows(Y_proj)
    return Y_proj, new_rank


# perform fixed point iteration on cvxpy variable X, where X is a solution of prob
def fixed_point_iteration(prob, X, shift=None, is_complex=False, returns_path=False, tol=1e-4, verbose=False,
                          solver=cp.MOSEK):
    n = X.shape[0]
    if shift is None:
        shift = np.zeros(n)
    X_path = None
    if returns_path:
        X_path = []

    if is_complex:
        prev_X = cp.Parameter((n, n), hermitian=True, value=X.value)
        iteration_obj = cp.real(cp.trace(X @ cp.real(prev_X + shift)))
    else:
        prev_X = cp.Parameter((n, n), symmetric=True, value=X.value)
        iteration_obj = cp.trace(X @ (prev_X + shift))

    def print_iteration_info(phase_keyword, prob, X, verbose=True):
        print("%s objective: %f" % (phase_keyword, prob.objective.value))
        if verbose:
            print("%s eigenvalues:" % (phase_keyword))
            print(np.linalg.eigvalsh(X.value))

    iteration_prob = cp.Problem(cp.Maximize(iteration_obj), prob.constraints)
    terminate = False
    n_iter = 0
    print_iteration_info("initial", prob, X)
    while not terminate:
        iteration_prob.solve(solver=solver)
        n_iter += 1
        if returns_path:
            X_path.append(X.value)
        if np.linalg.norm(X.value - prev_X.value) < tol:
            terminate = True
        else:
            print_iteration_info("current", prob, X, verbose)
            prev_X.value = X.value
    print_iteration_info("fixed point", prob, X)
    print("iterations: ", n_iter)
    if returns_path:
        return np.array(X_path)


# load n vertices of a graph as a networkx Graph
# if n == 0, load the entire graph
# if random == True, then first_node doesn't have any effect
def load_graph(graph_file, type, n=0, first_node=0, random=False):
    data_path = "dat/"
    with open(data_path + graph_file) as f:
        if type == 0:
            # toruspm
            next(f, '')  # skip first line
            G = nx.read_weighted_edgelist(f, nodetype=int, encoding="utf-8")
            G = nx.convert_node_labels_to_integers(G)
        elif type == 1:
            # matrix market
            import scipy as sp
            import scipy.io  # for mmread() and mmwrite()
            G = nx.from_scipy_sparse_array(sp.io.mmread(f))

    if n > 0:
        if random:
            first_vertex = np.floor(np.random.default_rng().random() * (len(G) - n - 1)).astype(int)
            G = G.subgraph(list(G.nodes)[first_vertex:first_vertex + n])
        else:
            G = G.subgraph(list(G.nodes)[first_node: (first_node + n)])
        assert len(G) == n
    nx.draw(G, nx.circular_layout(G))
    return G


def build_enriched_supergraph(G, treewidth_algorithm_idx=0):
    # use the chosen algorithm to compute the approximate minimal tree decomposition
    treewidth_algorithms_list = [nx.algorithms.approximation.treewidth_min_degree,
                                 nx.algorithms.approximation.treewidth_min_fill_in]  # nx.junction_tree
    treewidth, tree_decomp = treewidth_algorithms_list[treewidth_algorithm_idx](G)
    print("Treewidth: %s" % treewidth)

    G_bar = G.copy()
    next_idx = G_bar.number_of_nodes()  # keep track of index of the next redundant vertex to be added
    T_bar = tree_decomp.copy()

    # add redundant vertices
    for bag in tree_decomp.nodes:
        if len(bag) < treewidth + 1:
            new_bag = bag.union(frozenset(range(next_idx, next_idx + treewidth + 1 - len(bag))))
            T_bar = nx.relabel_nodes(T_bar, {bag: new_bag})
            next_idx += treewidth + 1 - len(bag)
    G_bar.add_nodes_from(range(G_bar.number_of_nodes(), next_idx))

    # iterate through leaves of T_tilde and add edges
    T_tilde = T_bar.copy()
    while T_tilde.number_of_nodes() > 1:
        for bag in T_tilde.nodes:
            if T_tilde.degree(bag) == 1:  # the current bag is a leaf
                parent = list(T_tilde.neighbors(bag))[0]
                Os = sorted(list(bag.difference(parent)))
                Ws = sorted(list(parent.difference(bag)))
                G_bar.add_edges_from([(Os[i], Ws[i]) for i in range(len(Os))])
                T_tilde.remove_node(bag)
                break

    return G_bar

# verifies X is exact and recovers the incidence vector from X
def recover_incidence_vector(X, type, tol=1e-6):
    n = X.shape[0]
    x = None
    if type == "lovasz":
        x = decompose_psd(X)
        assert x.shape[1] == 1
        x = x * 1 / np.min(x[x > tol])
    elif type == "grotschel":
        x = np.diag(X)
    elif type == "benson":
        x = decompose_psd(X, tol)
        assert x.shape[1] == 1  # needs to be rank 1
        x = x.reshape(1, -1)[0]
        x *= x[n - 1]  # the last entry is the sign corresponding to the stable set
        x = x[0:n-1]  # remove the last entry
        x[x < 0] = 0  # set -1 to 0
        n -= 1

    assert all(np.isclose(x[i], 0) or np.isclose(x[i], 1) for i in range(n))

    return np.round(x)
