import numpy as np
import matplotlib.pyplot as plt


# decompose a positive semidefinite matrix X = YY*
def decompose_psd(X, tol=1e-7):
    n = X.shape[0]
    eigen_val, eigen_vec = np.linalg.eigh(X)
    non_neg_eig_idx = eigen_val >= tol
    return eigen_vec[:, non_neg_eig_idx] @ np.diag(np.sqrt(eigen_val[non_neg_eig_idx]))


# remove the small eigenvalues (all eigenvalues less than tol) of a PSD matrix (so all negative eiganvalues will be
# removed)
def remove_small_eigenvalues(X, tol=1e-6):
    eigen_val, eigen_vec = np.linalg.eigh(X)
    eigen_val = np.real(eigen_val)
    idx_to_keep = eigen_val >= tol
    return eigen_vec[:, idx_to_keep] @ np.diag(eigen_val[idx_to_keep]) @ eigen_vec[:, idx_to_keep].T


# normalize the rows of matrix X
def normalize_rows(X):
    return X / np.linalg.norm(X, axis=1)[:, np.newaxis]


# get the eigenvalues and eigenvectors of PSD X, removing the delta_rank smallest eigenvalues
def eigh_proj(X, delta_rank):
    assert X.shape[0] > delta_rank
    eigen_val, eigen_vec = np.linalg.eigh(X)
    eigen_val = np.real(eigen_val)
    # sort by eigenvalues, from the largest to the smallest
    idx = eigen_val.argsort()[::-1]
    eigen_val = eigen_val[idx]
    eigen_vec = eigen_vec[:, idx]
    return eigen_val, eigen_vec


# eigen-project PSD X to decrease its rank by delta_rank
def eigen_proj(X, delta_rank):
    assert X.shape[0] > delta_rank
    current_rank = np.linalg.matrix_rank(X, tol=1e-9)
    target_rank = current_rank - delta_rank
    X_proj = X
    new_rank = current_rank
    if target_rank >= 1:
        new_rank = target_rank
        eigen_val, eigen_vec = eigh_proj(X, delta_rank)

        X_proj = eigen_vec[:, range(new_rank)] @ np.diag(eigen_val[range(new_rank)]) @ eigen_vec[:, range(new_rank)].T
    return X_proj, new_rank


def plot_matrix_heatmap(X):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(X, cmap='hot')
    plt.show()