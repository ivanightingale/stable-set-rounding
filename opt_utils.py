import numpy as np
import cvxpy as cp
import networkx as nx
from linalg_utils import normalize_rows, eigh_proj, decompose_psd


# round each row of Y to the unit annulus, or project onto the annulus between min_r and max_r
def hyperplane_rounding(Y, cost, min_r=1, max_r=1, n_iter=100):
    min_cost = np.Inf
    best_x = None
    best_x_temp = None
    d = Y.shape[1]
    rng = np.random.default_rng()
    for i in range(n_iter):
        r = rng.standard_normal((d, 1))
        x_temp = Y @ r
        x = np.sign(x_temp)

        cost_val = cost(x)
        if cost_val < min_cost:
            best_x = x
            min_cost = cost_val
            best_x_temp = x_temp
    return best_x, min_cost, x_temp


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


def find_center_point(prob, X):
    prob_center = cp.Problem(cp.Maximize(cp.log_det(X)), prob.constraints)
    prob.solve(solver=cp.SCS)


# perform fixed point iteration on cvxpy variable X, where X is a solution of prob
def fixed_point_iteration(prob, X, shift=None, returns_path=False, tol=1e-4, verbose=False,
                          solver=cp.MOSEK):
    n = X.shape[0]
    if shift is None:
        shift = np.zeros(n)
    X_path = None
    if returns_path:
        X_path = []

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
def load_graph(graph_file, type, n=0, first_node=0, random=False, display=False):
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

    if display:
        nx.draw(G, nx.circular_layout(G))
    return G


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
        x = x[0:n - 1]  # remove the last entry
        x[x < 0] = 0  # set -1 to 0
        n -= 1

    assert all(np.isclose(x[i], 0) or np.isclose(x[i], 1) for i in range(n))

    return np.round(x)


def lovasz_sdp(G):
    n = G.number_of_nodes()
    Z = cp.Variable((n, n), PSD=True)
    J = cp.Parameter((n, n), symmetric=True, value=np.ones((n, n)))
    constraints = [cp.trace(Z) == 1]
    constraints += [Z[i][j] == 0 for (i, j) in G.edges]
    prob = cp.Problem(cp.Maximize(cp.trace(J @ Z)), constraints)
    prob.solve(solver=cp.SCS, eps_abs=1e-6, eps_rel=1e-6, verbose=True)
    print(prob.value)
    return prob, Z


def grotschel_sdp(G):
    n = G.number_of_nodes()
    X = cp.Variable((n, n), symmetric=True)
    x = cp.Variable((n, 1), nonneg=True)
    X_plus = cp.bmat([[cp.Constant([[1]]), x.T], [x, X]])
    e = cp.Parameter(n, value=np.ones(n))
    constraints = [X_plus >> 0]
    constraints += [X[i][i] == x[i] for i in range(n)]
    constraints += [X[i][j] == 0 for (i, j) in G.edges]
    # prob_grotschel = cp.Problem(cp.Maximize(e @ x), constraints_grotschel)  # slightly different from result of maximizing tr(X)
    prob = cp.Problem(cp.Maximize(cp.trace(X)), constraints)
    prob.solve(solver=cp.SCS, eps_abs=1e-6, eps_rel=1e-6, verbose=True)
    print(prob.value)
    return prob, X, x, X_plus


def benson_sdp(G):
    n = G.number_of_nodes()
    V = cp.Variable((n + 1, n + 1), PSD=True)
    constraints = [V[i][i] == 1 for i in range(n + 1)]
    constraints += [V[i][i] + V[j][j] + V[n][n] + 2 * (V[i][j] + V[i][n] + V[j][n]) == 1 for (i, j) in G.edges]
    cost_mat = np.block([[0.5 * np.eye(n), np.array([[0.25]] * n)], [np.array([[0.25]] * n).T, 0]])
    prob = cp.Problem(cp.Maximize(cp.trace(cost_mat @ V)), constraints)
    prob.solve(solver=cp.SCS, eps_abs=1e-6, eps_rel=1e-6, verbose=True)
    # prob.solve(solver=cp.MOSEK)
    print(prob.value)
    return prob, V


def greedy_stable_set_rounding(X, G, n_iter=100):
    n = G.number_of_nodes()
    rng = np.random.default_rng()
    max_val = -np.Inf
    max_x = None
    for i in range(n_iter):
        remaining_vertices = np.array(range(n))
        greedy_x = np.zeros(n)
        p = np.diag(X) / np.sum(np.diag(X))
        while len(remaining_vertices) > 0:
            current_vertex = rng.choice(remaining_vertices, p=p)
            greedy_x[current_vertex] = 1
            idx_to_keep = [v for v in remaining_vertices if v != current_vertex and v not in G.neighbors(current_vertex)]
            remaining_vertices = remaining_vertices[idx_to_keep]
            p = p[idx_to_keep]
            p = p / np.sum(p)

        current_val = np.sum(greedy_x)
        if current_val > max_val:
            max_val = current_val
            max_x = greedy_x

    return max_x


# sample points in the feasible region of an SDP. Read existing samples from file if possible. Write old and new samples
# to file if necessary
def sdp_sampling(prob, X, sdp_type, folder, graph_file, n_iter=0, write_to_file=True):
    n = X.shape[0]
    samples_path = folder + "/dat/samples/%s_samples_%s_%d.npy" % (sdp_type, graph_file, n)
    try:
        samples = np.load(samples_path)
    except:
        samples = None
    rng = np.random.default_rng()
    C = cp.Parameter((n, n), symmetric=True)
    sampling_prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), prob.constraints)

    new_samples = [[]] * n_iter
    for i in range(n_iter):
        # generate random symmetric matrix value for C
        A = rng.uniform(-1, 1, (n, n))
        C.value = A.T + A
        sampling_prob.solve()
        new_samples[i] = X.value

    if n_iter > 0:
        if samples is None:
            samples = np.array(new_samples)
        else:
            samples = np.append(samples, np.array(new_samples), axis=0)

    if write_to_file:
        np.save(samples_path, samples)

    return samples
