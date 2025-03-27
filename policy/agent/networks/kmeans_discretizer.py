import tqdm
import torch

class KMeansDiscretizer:
    """
    Simplified and modified version of KMeans algorithm from sklearn.

    Code borrowed from https://github.com/notmahi/miniBET/blob/main/behavior_transformer/bet.py
    """

    def __init__(
        self,
        num_bins: int = 100,
        kmeans_iters: int = 50,
    ):
        super().__init__()
        self.n_bins = num_bins
        self.kmeans_iters = kmeans_iters

    def fit(self, input_actions: torch.Tensor) -> None:
        self.bin_centers = KMeansDiscretizer._kmeans(
            input_actions, nbin=self.n_bins, niter=self.kmeans_iters
        )

    @classmethod
    def _kmeans(cls, x: torch.Tensor, nbin: int = 512, niter: int = 50):
        """
		Function implementing the KMeans algorithm.

		Args:
			x: torch.Tensor: Input data - Shape: (N, D)
			nbin: int: Number of bins
			niter: int: Number of iterations
		"""
          
		# TODO: Implement KMeans algorithm to cluster x into nbin bins. Return the bin centers - shape (nbin, x.shape[-1])
          
        N, D = x.shape

        indices = torch.randperm(N)[:nbin]
        bin_centers = x[indices].clone()

        for _ in range(niter):
            # 1. Compute distances to each center
            dist = (x.unsqueeze(1) - bin_centers.unsqueeze(0)) ** 2
            dist = dist.sum(dim=-1)

            # 2. Assign each sample to the closest center
            cluster_ids = dist.argmin(dim=1)  # shape (N,)

            # 3. Update each center
            for k in range(nbin):
                mask = (cluster_ids == k)
                if mask.any():
                    bin_centers[k] = x[mask].mean(dim=0)

        return bin_centers

