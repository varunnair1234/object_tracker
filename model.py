import torch

class AppearanceModel():
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.singular_values = None
        self.buffer = []
        self.buffer_size = 30
    
    def initialize(self, patch):
        self.mean = patch

    def score(self, patch):
        mean_center = patch - self.mean         # this is a 1024 dimensional vector
        return torch.norm(mean_center)          # torch.norm computs the magnitude of vector mean_center

    def update(self, patch):
        buffer = self.buffer
        buffer.append(patch)
        if len(buffer) > self.buffer_size:
            buffer.pop(0)
        #PCA computation
        if len(buffer) >= self.n_components + 1:
            stacked = torch.stack(buffer)
            self.mean = torch.mean(stacked, dim=0)
            centered = stacked - self.mean
            U, S, Vt = torch.linalg.svd(centered, full_matrices = False)            #   U = how much of PC1 is in this patch, how much of PC2 is in it
            self.components = Vt[:self.n_components]
            self.singular_values = S[:self.n_components]