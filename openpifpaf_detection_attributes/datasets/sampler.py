import torch


class RegularSubSampler(torch.utils.data.sampler.Sampler[int]):
    """Subsampler for video datasets.

    Images are subsampled with a regular step within each subepoch.
    Each epoch on the sampler (subepoch of the full dataset) corresponds to a
    different subset of the dataset, until all examples are seen.

    Args:
        data_size (int): Size of dataset to subsample.
        subepochs (int): Number of subepochs corresponding to the full dataset.
        shuffle (bool): Randomize order of eamples.
    """

    def __init__(self,
                 data_size: int,
                 subepochs: int = 1,
                 shuffle: bool =  False):
        self.data_size = data_size
        assert subepochs > 0
        self.subepochs = subepochs
        self.shuffle = shuffle

        self._subepoch = None
        self._subepoch_idx = None


    @property
    def num_samples(self):
        return self.data_size // self.subepochs


    def _new_epoch(self):
        self._subepoch_idx = 0
        if self.shuffle:
            self._subepoch_order = torch.randperm(self.subepochs)
        else:
            self._subepoch_order = torch.arange(self.subepochs)


    def _new_subepoch(self):
        if self._subepoch_idx is None:
            self._subepoch_idx = self.subepochs
        self._subepoch_idx += 1
        if self._subepoch_idx >= self.subepochs:
            self._new_epoch()

        self._subepoch = self._subepoch_order[self._subepoch_idx]


    def __iter__(self):
        self._new_subepoch()
        if self.shuffle:
            example_order = torch.randperm(self.num_samples)
        else:
            example_order = torch.arange(self.num_samples)
        example_order *= self.subepochs
        example_order += self._subepoch
        yield from example_order.tolist()


    def __len__(self):
        return self.num_samples
