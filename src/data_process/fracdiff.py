"""
Module to define fractional differentiation
"""

import numpy as np


class Fracdiff:
    """
    Fractional differentiator
    """

    def __init__(self):
        self.fd_weights = None  # frac diff weights

    def get_weights(self, window_length: int, order: float) -> np.array:
        """
        Compute and assign fractional differentiation weights
        """
        weights = np.ones(shape=window_length, dtype=np.float32)
        for i in range(1, window_length):
            weights[i] = -weights[i - 1] * (order - i + 1) / i

        self.fd_weights = weights
