import torch
import torch.nn.functional as F
from torch import nn
import math
from rms_norm import RMSNorm
from multiway_network import MultiwayWrapper

def theta_shift(x, sin, cos):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rotated = torch.stack((-x2, x1), dim=-1)
    return x_rotated.flatten(-2)

class MultiScaleRetention(nn.Module):
    def __init__(self, input_dim, value_dim=512, num_heads=8, forecast_horizon=120, gate_fn="gelu", layernorm_eps=1e-5, multiway=False):
        super(MultiScaleRetention, self).__init__()
        self.input_dim = input_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.forecast_horizon = forecast_horizon
        self.head_dim = value_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        if gate_fn == "swish":
            self.gate_fn = F.silu
        elif gate_fn == "gelu":
            self.gate_fn = F.gelu
        else:
            raise ValueError("Unsupported gate function type")

        # Initialize multiway projections
        args = {'multiway': multiway, 'layernorm_eps': layernorm_eps}
        self.q_proj = MultiwayWrapper(args, nn.Linear(input_dim, self.num_heads * self.head_dim, bias=False))
        self.k_proj = MultiwayWrapper(args, nn.Linear(input_dim, self.num_heads * self.head_dim, bias=False))
        self.v_proj = MultiwayWrapper(args, nn.Linear(input_dim, self.num_heads * self.head_dim, bias=False))
        self.out_proj = MultiwayWrapper(args, nn.Linear(self.num_heads * self.head_dim, forecast_horizon, bias=False))
        
        # Ensure the dimension for RMSNorm matches the expected output dimension from out_proj
        output_dim = self.num_heads * self.head_dim  # Adjust this if it doesn't match your model's output
        self.norm = RMSNorm(output_dim, eps=layernorm_eps)

    # Other methods remain unchanged



    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def recurrent_forward(self, qr, kr, v, incremental_state=None):
        bsz, num_heads, seq_len, head_dim = qr.size(0), self.num_heads, qr.size(1), self.head_dim
        qr = qr.view(bsz, num_heads, seq_len, head_dim)
        v = v.view(bsz, num_heads, seq_len, head_dim)
        kr = kr.view(bsz, num_heads, seq_len, head_dim)
        kv = kr * v  # Element-wise multiplication

        if incremental_state is None:
            incremental_state = {'prev_key_value': torch.zeros_like(kv), 'scale': torch.ones(bsz, num_heads, 1, head_dim, device=qr.device)}
        
        prev_kv = incremental_state['prev_key_value']
        kv = 0.9 * prev_kv + 0.1 * kv
        incremental_state['prev_key_value'] = kv

        output = torch.sum(qr * kv, dim=2)
        scale = incremental_state['scale'].expand_as(kv).sum(dim=2, keepdim=True)
        output = output / scale.squeeze(2)
        return output, incremental_state
    
    def chunk_recurrent_forward(self, qr, kr, v, incremental_state):
        bsz, num_heads, seq_len, head_dim = v.size()
        chunk_len = 16
        if seq_len % chunk_len != 0:
            raise ValueError(f"Sequence length ({seq_len}) must be divisible by chunk length ({chunk_len}).")

        num_chunks = seq_len // chunk_len
        qr = qr.view(bsz, num_heads, num_chunks, chunk_len, head_dim).transpose(1, 2)
        kr = kr.view(bsz, num_heads, num_chunks, chunk_len, head_dim).transpose(1, 2)
        v = v.view(bsz, num_chunks, chunk_len, num_heads, head_dim).transpose(2, 3)
        kr_t = kr.transpose(-2, -1)
        qk_mat = torch.matmul(qr, kr_t)
        mask = torch.ones(bsz, num_chunks, num_heads, chunk_len, chunk_len, device=v.device)
        qk_mat *= mask
        inner_scale = qk_mat.sum(-1, keepdim=True).clamp(min=1)
        qk_mat /= inner_scale
        inner_output = torch.matmul(qk_mat, v)
        kv = kr_t @ v
        kv = kv.view(bsz, num_heads, -1, head_dim).transpose(1, 2)  # reshape kv to match dimensions [batch, seq, heads, dim]

        # Flatten the dimensions correctly for RMSNorm
        kv_flat = kv.contiguous().view(bsz, -1)
        # print("Shape of kv before normalization:", kv_flat.size())

        if kv_flat.size(1) != 96:
            # print("Incorrect dimension for RMSNorm, adjusting...")
            # Adjust dimension to fit RMSNorm expected size
            kv_flat = kv_flat[:, :96]

        output = self.norm(kv_flat)
        return output


    # def chunk_recurrent_forward(self, qr, kr, v, incremental_state):
    #     bsz, num_heads, seq_len, head_dim = v.size()
    #     chunk_len = 16

    #     if seq_len % chunk_len != 0:
    #         raise ValueError(f"Sequence length ({seq_len}) must be divisible by chunk length ({chunk_len}).")

    #     num_chunks = seq_len // chunk_len
    #     qr = qr.view(bsz, num_heads, num_chunks, chunk_len, head_dim).transpose(1, 2)
    #     kr = kr.view(bsz, num_heads, num_chunks, chunk_len, head_dim).transpose(1, 2)
    #     v = v.view(bsz, num_chunks, chunk_len, num_heads, head_dim).transpose(2, 3)

    #     kr_t = kr.transpose(-2, -1)
    #     qk_mat = torch.matmul(qr, kr_t)

    #     mask = torch.ones(bsz, num_chunks, num_heads, chunk_len, chunk_len, device=v.device)
    #     qk_mat *= mask

    #     inner_scale = qk_mat.sum(-1, keepdim=True).clamp(min=1)
    #     qk_mat /= inner_scale

    #     inner_output = torch.matmul(qk_mat, v)
    #     kv = kr_t @ v
    #     print("Shape of kv before view:", kv.size())  # Debug output

    #     kv = kv.contiguous().view(bsz, num_heads, -1, head_dim)  # Flatten chunks into sequence dimension
    #     print("Shape of kv after reshape:", kv.size())  # Debug output

    #     output = self.norm(kv.view(bsz, -1))  # Flatten all dimensions except batch for RMSNorm
    #     print("Output shape before normalization:", output.shape)

    #     kv_recurrent = []
    #     cross_scale = []

    #     for i in range(num_chunks):
    #         if i < kv.size(2):  # Check if index is within the new shape
    #             current_kv = kv[:, :, i, :]
    #             kv_recurrent.append(current_kv)
    #         else:
    #             print(f"Index {i} out of bounds for kv with shape {kv.shape}")

    #     if kv_recurrent:
    #         kv_recurrent = torch.stack(kv_recurrent, dim=2)  # Adjust dimension for stacking
    #         output = self.compute_output(kv_recurrent, qr, inner_scale)
    #         return output
    #     else:
    #         print("No valid kv slices to stack, check earlier operations.")
    #         return torch.zeros(bsz, seq_len, self.head_dim * self.num_heads, device=qr.device)





    def forward(self, x, use_chunk=False, incremental_state=None):
        batch_size, seq_length, _ = x.shape
        sin, cos = self.generate_positional_encodings(seq_length, batch_size, x.device)
        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x) * self.scaling
        v = self.v_proj(x)
        v = v.view(batch_size, self.num_heads, seq_length, self.head_dim)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        if use_chunk:
            output, _ = self.chunk_recurrent_forward(qr, kr, v, incremental_state)  # Unpack the tuple here
        else:
            output, _ = self.recurrent_forward(qr, kr, v, incremental_state)  # Unpack the tuple here

        output = output.view(batch_size, -1)
        output = self.norm(output)
        output = self.gate_fn(output)
        output = self.out_proj(output)
        return output


    def generate_positional_encodings(self, seq_length, batch_size, device):
        position = torch.arange(seq_length, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.head_dim, 2, device=device) * -(math.log(10000.0) / self.head_dim))
        sin = torch.sin(position * div_term)
        cos = torch.cos(position * div_term)
        return sin.repeat(batch_size, 1, 1), cos.repeat(batch_size, 1, 1)



