import torch
import torch.nn as nn
import numpy as np

class Scrambler:
    def __init__(self, size=256):
        # Create a fixed permutation key
        # In a real app, save 'perm' to a file to use it later!
        self.size = size
        self.pixel_count = size * size
        rng = np.random.default_rng(seed=42) # Fixed seed = Fixed Key
        self.perm = torch.from_numpy(rng.permutation(self.pixel_count)).long()
        self.inv_perm = torch.argsort(self.perm)

    def scramble(self, image_tensor):
        # Flatten, shuffle, reshape
        b, c, h, w = image_tensor.shape
        flat = image_tensor.view(b, c, -1)
        shuffled = flat[:, :, self.perm]
        return shuffled.view(b, c, h, w)

    def descramble(self, image_tensor):
        # Flatten, un-shuffle, reshape
        b, c, h, w = image_tensor.shape
        flat = image_tensor.view(b, c, -1)
        unshuffled = flat[:, :, self.inv_perm]
        return unshuffled.view(b, c, h, w)
