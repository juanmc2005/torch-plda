import torch
from scipy.linalg import eigh
import utils


def scatter_matrices(X: torch.Tensor, y: torch.Tensor, device: str):
    """
    Compute within-class and between-class scatter matrices according to the algorithm described in:
        https://ravisoji.com/assets/papers/ioffe2006probabilistic.pdf
        (Figure 2, p537)
    :param X: a Float matrix of training vectors of size (N, d), where d is the vector dimension
    :param y: a Long vector of class labels
    :param device: a PyTorch device where to execute the operations
    :return: (S_w, S_b) where:
        S_w is the within-class scatter matrix of size (d, d)
        S_b is the between-class scatter matrix of size (d, d)
    """
    # Number of examples and dimension of each example vector
    N, d = X.size(0), X.size(1)
    # Get the unique list of classes
    K = torch.unique(y, sorted=True)
    # Calculate training set mean
    m = torch.mean(X, dim=0)
    # Initialize within-class and between-class scatter matrices to 0
    S_w, S_b = torch.zeros(d, d).to(device), torch.zeros(d, d).to(device)
    # For every class k
    for k in K:
        # Select x_i such that y_i == k
        C_k = utils.select_class(X, y, k)
        # Number of examples of class k
        n_k = C_k.size(0)
        # Calculate the class mean
        m_k = torch.mean(C_k, dim=0)
        # Calculate difference between each class example and the class mean
        diff_X_k = C_k - m_k.expand(C_k.size(0), -1)
        # Calculate the difference between the class mean and the general mean
        diff_mk_m = m_k - m
        # Multiply X and m_k difference my itself (transpose goes first because PyTorch uses row vectors)
        S_w_k = torch.matmul(diff_X_k.transpose(0, 1), diff_X_k)
        # Multiply means' difference by itself (unsqueeze(1) gives a column vector and unsqueeze(0) a row vector)
        S_b_k = n_k * torch.matmul(diff_mk_m.unsqueeze(1), diff_mk_m.unsqueeze(0))
        # Add this class' matrices
        S_w += S_w_k
        S_b += S_b_k
    # Divide by the number of examples
    S_w /= N
    S_b /= N
    return S_w, S_b


def plda(X: torch.Tensor, y: torch.Tensor, device: str):
    """
    Optimizes a PLDA model according to the algorithm described in:
        https://ravisoji.com/assets/papers/ioffe2006probabilistic.pdf
        (Figure 2, p537)
    :param X: a Float matrix of training vectors of size (N, d), where d is the vector dimension
    :param y: a Long vector of class labels
    :param device: a PyTorch device where to execute the operations
    :return: (m, A^-1, Psi), where:
        m is the mean of x_i vectors
        A^-1 is the inverse of the A diagonal matrix, which we use for predictions
        Psi is the Psi diagonal matrix that determines the most important features (non-zero entries)
    """
    # Number of examples and dimension of each example vector
    N, d = X.size(0), X.size(1)
    # Number of unique classes
    K = torch.unique(y).size(0)
    # Calculate training set mean
    m = torch.mean(X, dim=0)
    # Calculate within-class and between-class matrices
    S_w, S_b = scatter_matrices(X, y, device)
    # Find column eigenvectors W
    # FIXME this does not depend on PyTorch so we cannot use GPU
    _, W = eigh(S_b.cpu().numpy(), S_w.cpu().numpy())
    W = torch.tensor(W).to(device)
    WT = W.transpose(0, 1)
    # Calculate Lambda matrices W^T * S_w/b * W
    Lambda_b = torch.matmul(torch.matmul(WT, S_b), W).diagonal().diag()
    Lambda_w = torch.matmul(torch.matmul(WT, S_w), W).diagonal().diag()
    # Calculate n
    n = N / K
    # Calculate A matrix
    nfactorA = n / (n - 1)
    A = torch.inverse(WT) * ((nfactorA * Lambda_w) ** .5)
    # Calculate Psi matrix
    nfactorPsi = (n - 1) / n
    Psi = torch.clamp(nfactorPsi * (Lambda_b / Lambda_w) - 1 / n, 0)
    # Set m as a column vector
    # Inverse A as it is needed later to compute predictions
    # Select Psi's diagonal, as the rest is not needed
    return m.unsqueeze(1), A.inverse(), Psi.diagonal()


def plda_encode(batch: torch.Tensor, m: torch.Tensor,
                inv_A: torch.Tensor, latent_feat_idx: torch.Tensor) -> torch.Tensor:
    """
    Encode a batch of vectors in the latent space
    :param batch: a Float matrix of size (B, d), where d is the vector dimension
    :param m: a Float mean vector of size d, obtained from a fit PLDA model
    :param inv_A: a Float inversed A matrix of size (d, d), obtained from a fit PLDA model
    :param latent_feat_idx: a Long vector of size d', indicating the indices of important features of d.
        d' is the dimension of the latent space
        the indices must be in ascending order
    :return: a Float matrix of size (B, d'), the batch vectors in the latent space
    """
    # Transpose `batch` because PyTorch uses row vectors by default
    u = torch.matmul(inv_A, batch.transpose(0, 1) - m)
    # Transpose `u` so we can interpret as (batch_size, d)
    u = u.transpose(0, 1)
    # Select only valuable dimensions for the latent space
    return torch.index_select(u, 1, latent_feat_idx)
