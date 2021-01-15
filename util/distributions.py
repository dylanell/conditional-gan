"""
Custom distribution function classes.
"""

import torch
from torch.distributions.categorical import Categorical


class MixedMetaMultiHotCategorical():
    def __init__(self, batch_size, num_class, pan=0.5):
        """
        Combines a geometric distribution with bernoulli success prob 'p'
        with the reverse of itself with exponential smoothing parameter 'pan'
        controlling the weight of the reverse geometric distribution and
        rescaling to result in a valid mixed probability distribution. Result
        is a distibution with center of mass towards 1 for 'pan' close to 0
        (i.e. higher tendency to sample lower numbers), center of mass towards
        'num_class' for 'pan' close to 1 (i.e. higher tendency to sample
        numbers closer to 'num_class'), and approximately uniform distribution
        on [1, num_class] for 'pan' close to 0.5. Values sampled from this
        distribution are used to determine how many unique labels to uniformly
        sample for each instance of a randomly sampled multi-hot class label,
        therefore 'pan' close to 0 results in higher tendency to sample one-hot
        'multi-labels' (consisting of one class and low 'pan-classness') and
        higher 'pan' results in higher tendency to sample all-hot 'multi-
        labels' (consisting of all 'num_class' classes and high 'pan-
        classness').
        """

        # bernoulli trial success probability p in (0, 1]
        # as 'pan'-> 0 or 1, want p -> 1 for sharp geometric distribution
        # when 'pan' -> 0.5, want p -> 0 for 'flat' geometric distribution
        # p is limited to (0, 1] to avoid p=0 and p>1 cases
        min_p, max_p = (1e-4, 1)
        p = torch.abs(
            torch.tensor((2 * (max_p - min_p) * (pan - 0.5)) + min_p))

        # geometric distibution from 1-num_class
        one_hot_probs = torch.tensor(
            [p * ((1 - p)**i) for i in range(num_class)])

        # reversed geometric distibution from 1-num_class
        multi_hot_probs = torch.tensor(
            [p * ((1 - p)**i) for i in range(num_class)][::-1])

        # mix probabilities with exponential smoothing with 'pan'
        # pan=0, low 'pan-classness' therefore favor one_hot_probs
        # pan=1, high 'pan-classness' therefore favor multi_hot_probs
        # rescale mixed probs to valid probability distribution while adding
        # small value to avoid
        mixed_probs = ((1 - pan) * one_hot_probs) + (pan * multi_hot_probs)
        probs = mixed_probs / torch.sum(mixed_probs)

        # construct a categorical distribution with probability values
        k_dist = Categorical(probs.unsqueeze(0).repeat(batch_size, 1))

        self._batch_size = batch_size
        self._num_class = num_class
        self._k_dist = k_dist

    def sample(self):
        # sample number of multi-class labels for each batch
        k_samples = self._k_dist.sample()+1

        # randomly select label indices for each multi-class label batch
        idx_samples = [
            torch.randperm(self._num_class)[:k_val] for k_val in k_samples]

        # construct num_sample multi-hot vectors from sampled indices
        multi_hots = torch.zeros(self._batch_size, self._num_class)
        for i, (idx_s, k_s) in enumerate(zip(idx_samples, k_samples)):
            multi_hots[i, idx_s] = 1/k_s

        return multi_hots
