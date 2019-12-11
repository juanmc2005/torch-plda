import torch
import warnings
import libplda as lib


# TODO add PLDA Scoring for speech processing applications
class PLDA:
    """
    Probabilistic Linear Discriminant Analysis as described in:
        https://ravisoji.com/assets/papers/ioffe2006probabilistic.pdf
    """

    def __init__(self, latent_space_dim: int = None, device: str = 'cpu'):
        """
        :param latent_space_dim: the number of dimensions in the latent space.
            This value might be overriden if not enough features are discoverded by the algorithm
        :param device: a PyTorch device where to execute the operations
        """
        self.latent_dim = latent_space_dim
        self.device = device
        self.m, self.inv_A, self.Psi, self.latent_idx = None, None, None, None

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fits the PLDA model according to the algorithm described in:
            https://ravisoji.com/assets/papers/ioffe2006probabilistic.pdf
            (Figure 2, p537)
        :param X: a Float matrix of training vectors of size (N, d), where d is the vector dimension
        :param y: a Long vector of class labels
        """
        with torch.no_grad():
            X, y = X.to(self.device), y.to(self.device)
            self.m, self.inv_A, self.Psi = lib.plda(X, y, self.device)
            # Update latent space dimension if not specified or not possible
            n_important_feats = (self.Psi > 0).sum().item()
            if self.latent_dim is not None and n_important_feats < self.latent_dim:
                warnings.warn(f"PLDA identified a latent space dimension of {n_important_feats}, "
                              f"but the user specified {self.latent_dim}. "
                              f"Setting latent space dimension to {n_important_feats}")
                self.latent_dim = n_important_feats
            elif self.latent_dim is None:
                self.latent_dim = n_important_feats
            # Get the ids of the `latent_dim` most important features detected by the model
            self.latent_idx = torch.argsort(self.Psi, descending=True)[:self.latent_dim].sort()[0]

    def __call__(self, batch) -> torch.Tensor:
        """
        Encode a batch of vector examples using the previously fit model
        :param batch: a Float matrix of vectors of size (n, d), where d is the vector dimension
        :return: the batch's latent representations. A Float matrix of vectors of size (n, latent_d)
        """
        with torch.no_grad():
            batch = batch.to(self.device)
            if self.latent_idx is None:
                raise AssertionError('You need to call `fit` before applying the model')
            # Transpose `batch` because PyTorch uses row vectors by default
            u = torch.matmul(self.inv_A, batch.transpose(0, 1) - self.m)
            # Transpose `u` so we can interpret as (batch_size, d)
            u = u.transpose(0, 1)
            # Select only valuable dimensions for the latent space
            return torch.index_select(u, 1, self.latent_idx)
