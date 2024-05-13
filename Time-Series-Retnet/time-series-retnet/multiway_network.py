# # Copyright (c) 2022 Microsoft
# # Licensed under The MIT License [see LICENSE for details]


import copy
import torch
import torch.nn as nn

def MultiwayWrapper(args, module, dim=1):
    """A wrapper for applying the Multiway functionality to a module."""
    if args.get('multiway', False):  # Correctly access the 'multiway' key in args
        return MultiwayNetwork(module, dim=dim)
    return module

class MultiwayNetwork(nn.Module):
    """A network that supports running different computations on split parts of the input."""
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.A = module  # Primary module
        self.B = copy.deepcopy(module)  # Secondary module, a deep copy of the primary
        self.B.reset_parameters()  # Reset parameters of the secondary module
        self.split_position = -1  # Default split position

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim
        )
        y1 = self.A(x1, **kwargs)
        y2 = self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)

class MutliwayEmbedding(MultiwayNetwork):
    """A specific case of MultiwayNetwork intended for use with embedding layers."""
    def __init__(self, modules, dim=1):
        assert len(modules) == 2, "MutliwayEmbedding expects exactly two modules."
        super().__init__(modules[0], dim=dim)
        self.B = modules[1]  # Overwrite the secondary module

# Example of defining a model that uses MultiwayWrapper
class ExampleModel(nn.Module):
    def __init__(self, embed_dim, value_dim, num_heads, multiway=False):
        super().__init__()
        args = {'multiway': multiway, 'layernorm_eps': 1e-5}
        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=False))
        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=False))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, value_dim, bias=False))
        self.out_proj = MultiwayWrapper(args, nn.Linear(value_dim, embed_dim, bias=False))

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        output = self.out_proj(v)
        return output

# # Use the model
# model = ExampleModel(embed_dim=128, value_dim=128, num_heads=4, multiway=True)
# print(model)


# import copy

# import torch
# import torch.nn as nn


# def MultiwayWrapper(args, module, dim=1):
#     if args.multiway:
#         return MultiwayNetwork(module, dim=dim)
#     return module


# def set_split_position(position):
#     def apply_fn(module):
#         if hasattr(module, "split_position"):
#             module.split_position = position

#     return apply_fn


# class MultiwayNetwork(nn.Module):
#     def __init__(self, module, dim=1):
#         super().__init__()
#         self.dim = dim
#         self.A = module
#         self.B = copy.deepcopy(module)
#         self.B.reset_parameters()
#         self.split_position = -1

#     def forward(self, x, **kwargs):
#         if self.split_position == -1:
#             return self.A(x, **kwargs)
#         if self.split_position == 0:
#             return self.B(x, **kwargs)
#         x1, x2 = torch.split(
#             x,
#             [self.split_position, x.size(self.dim) - self.split_position],
#             dim=self.dim,
#         )
#         # x1, x2 = x[:self.split_position], x[self.split_position:]
#         y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
#         return torch.cat([y1, y2], dim=self.dim)


# class MutliwayEmbedding(MultiwayNetwork):
#     def __init__(self, modules, dim=1):
#         super(MultiwayNetwork, self).__init__()
#         self.dim = dim
#         assert len(modules) == 2
#         self.A = modules[0]
#         self.B = modules[1]
#         self.split_position = -1