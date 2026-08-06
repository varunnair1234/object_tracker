import torch

class AppearanceModel():
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.singular_values = None
        self.n_samples = 0
        self.buffer = []
        self.buffer_size = 30
    
    def initialize(self, patch):
        self.mean = patch
        self.n_samples = 1

    def score(self, patch):
        mean_center = patch - self.mean         # this is a 1024 dimensional vector
        return torch.norm(mean_center)          # torch.norm computs the magnitude of vector mean_center

    def update(self, patch):
        buffer = self.buffer
        buffer.append(patch)
        if len(buffer) > self.buffer_size:
            buffer.pop(0)
        
