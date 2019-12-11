import torch


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
