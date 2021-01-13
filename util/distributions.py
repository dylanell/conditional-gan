"""
Custom distribution function classes.
"""

import torch

class MetaMultiHotCategorical():
    """
    Creates a meta-distribution parameterized by 'size', 'n', and 'k' to draw
    'size' samples of n-dimensional, 'up-to-k-multi-hot' categorical
    distributions. This is a probability distribution to draw samples of
    another probability distribution. Given 'n', the number of unique labels
    for the multi-hot categorical distributions, and 'k', the maximum number of
    labels each sample from the multi-hot categorical distribution can have, 
    the MetaMultiHotCategorical distribution will produce samples consisting of
    'size' vectors of multi-hot categorical distributions of dimension 'n' in
    which 1 to k indices (sampled uniformly) will be non-zero and scaled such
    that the sum of all values is 1 (so that it is a probability distribuion).
    For example, if size=1, n=4, and k=3, sampled multi-hot distributions could
    include [0, 1/2, 1/2, 0], [1, 0, 0, 0], and [1/3, 1/3, 1/3, 0], but never
    [0, 0, 0, 0], or [1/4, 1/4, 1/4, 1/4].
    """
    def __init__(self, size, n, k):
        self._size = size
        self._n = n
        self._k = k

    def sample(self):
        # construct 'size' randomly selected sets of 1 to k 'hot' # indices
        k_samples = torch.randint(
            low=1, high=self._k+1, size=(self._size,))
        idx_samples = [torch.randperm(self._n)[:k_val] for k_val in k_samples]

        # construct num_sample multi-hot vectors from sampled indices
        multi_hots = torch.zeros(self._size, self._n)
        for i, (idx_sample, k_sample) in enumerate(zip(idx_samples, k_samples)):
            multi_hots[i, idx_sample] = 1/k_sample

        return multi_hots
