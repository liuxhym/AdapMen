from typing import Optional, Generator

import abc

import numpy as np


class BaseBuffer(abc.ABC):

    def __init__(self):
        super(BaseBuffer, self).__init__()
        self.size = 0

    @abc.abstractmethod
    def generate_batch_data(self, indices: np.ndarray):
        ...

    def get_batch_generator_inf(self, batch_size: Optional[int], ranges=None) -> Generator:
        ranges = range(self.size) if ranges is None else ranges
        batch_size = batch_size or len(ranges)
        while True:
            indices = np.random.choice(ranges, replace=True, size=batch_size)
            yield self.generate_batch_data(indices)

    def get_batch_generator_epoch(self, batch_size: Optional[int], ranges=None, **kwargs) -> Generator:
        ranges = np.arange(self.size) if ranges is None else ranges
        range_len = len(ranges)
        batch_size = batch_size or len(ranges)
        np.random.shuffle(ranges)
        ranges = ranges[:(range_len // batch_size) * batch_size].reshape([-1, batch_size])
        for indices in ranges:
            yield self.generate_batch_data(indices)
