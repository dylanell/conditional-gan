"""
Custom distribution function classes.
"""

import torch
from torch.distributions.categorical import Categorical

class TailBoostedMetaMultiHotCategorical():
    def __init__(self, batch_size, num_class, pan=0.5):
        """
        Bernoulli success probability p to parameterize a bound-modified
        geometric-like distribution for sampling 'multi-class' labels.
        Computation of p is dependent on 'pan-classness' parameter 'pan'. When
        'pan' is 0.0, meaning no 'pan-classness', bernoulli success probabily
        will be 1, resulting in only 'single-class' labels (i.e. a 100%
        probability of selecting 1 class). When 'pan' is 1.0, meaning full 'pan-
        classness', bernoulli success probabily will be N^(-2), resulting in
        small but nearly uniform probabilities for selectiing 'multi-class'
        labels in the range 1-(N-1). The probability mass from the remaining
        unbounded tail of the bernoulli distribution is summed for the
        probability at x=N, resuting in a distribution that strongly favors
        selecting 'all-N-class' labels when 'pan' is 1.0. When pan is 0.5, this
        results in a more balanced distribution of selecting 'multi-class'
        labels in the range 1-N. Therefore as the value of pan increases from 0-
        1, the probability of selecting samples that simultaneously belong to
        more classes increases.
        """

        # compute bernoulli success probability from pan-classness strength
        p = num_class**(-2 * pan)

        # compute modified geometric distribution with p success probability
        # and adding cumulative probability at (N+1)-inf to probability at N to
        # create bounded distribution on [1-N]
        probs = torch.tensor([p * ((1 - p)**i) for i in range(num_class - 1)])
        probs = torch.cat([probs, 1 - torch.sum(probs, dim=0, keepdim=True)])

        # construct a categorical distribution with the probability values
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

        # bernoulli trial success probability
        p = 0.5

        # geometric distibution from 1-num_class
        one_hot_probs = torch.tensor(
            [p * ((1 - p)**i) for i in range(num_class)])

        # reversed geometric distibution from 1-num_class
        multi_hot_probs = torch.tensor(
            [p * ((1 - p)**i) for i in range(num_class)][::-1])

        # mix probabilities with exponential smoothing with 'pan'
        # pan=0, low 'pan-classness' therefore all one_hot_probs
        # pan=1, high 'pan-classness' therefore all multi_hot_probs
        # square probs to boost outliers
        # rescale to valid probability distribution with softmax
        probs = ((1 - pan) * one_hot_probs) + (pan * multi_hot_probs)
        probs = probs**2
        probs /= torch.sum(probs)

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
