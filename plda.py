import torch
from scipy.linalg import eigh


def select_class(X: torch.Tensor, y: torch.Tensor, k: int):
    """
    Select all vector examples from the training set matrix X that correspond to the class k
    :param X: a Float matrix of size (N, d), where d is the vector dimension
    :param y: a Long vector of class labels
    :param k: the class to select
    :return: a Float torch.Tensor matrix of size (N_k, d), where N_k is the number of examples of class k
    """
    indices = (y == k).nonzero().squeeze(1)
    return torch.index_select(X, 0, indices)


def scatter_matrices(X: torch.Tensor, y: torch.Tensor):
    """
    Compute within-class and between-class scatter matrices according to the algorithm described in:
        https://ravisoji.com/assets/papers/ioffe2006probabilistic.pdf
        (Figure 2, p537)
    :param X: a Float matrix of training vectors of size (N, d), where d is the vector dimension
    :param y: a Long vector of class labels
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
    S_w, S_b = torch.zeros(d, d), torch.zeros(d, d)
    # For every class k
    for k in K:
        # Select x_i such that y_i == k
        C_k = select_class(X, y, k)
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


def plda(X: torch.Tensor, y: torch.Tensor):
    """
    Optimizes a PLDA model according to the algorithm described in:
        https://ravisoji.com/assets/papers/ioffe2006probabilistic.pdf
        (Figure 2, p537)
    :param X: a Float matrix of training vectors of size (N, d), where d is the vector dimension
    :param y: a Long vector of class labels
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
    S_w, S_b = scatter_matrices(X, y)
    # Find column eigenvectors W
    _, W = eigh(S_b.numpy(), S_w.numpy())
    W = torch.tensor(W)
    WT = W.transpose(0, 1)
    # Calculate Lambda matrices W^T * S_w/b * W
    Lambda_b = torch.matmul(torch.matmul(WT, S_b), W).diagonal().diag()
    Lambda_w = torch.matmul(torch.matmul(WT, S_w), W).diagonal().diag()
    # Calculate n
    n = N / K
    # Calculate A matrix
    nfactorA = n / (n-1)
    A = torch.inverse(WT) * ((nfactorA * Lambda_w) ** .5)
    # Calculate Psi matrix
    nfactorPsi = (n-1) / n
    Psi = torch.clamp(nfactorPsi * (Lambda_b / Lambda_w) - 1/n, 0)
    # Inverse A as it is needed later to compute predictions
    return m, A.inverse(), Psi
