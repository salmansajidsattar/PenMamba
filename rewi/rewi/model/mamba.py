"""Small pure-PyTorch Mamba (selective state space) block and the BiMamba + BiLSTM decoder.

No mamba-ssm install is needed. The scan is a plain loop over time, which is fine
here because the encoder output is short (at most 1024 // 8 = 128 steps).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MambaBlock', 'MambaLSTM']


def selective_scan(u, dt, A, B, C, D):
    '''Sequential selective scan, always in float32.

    Args:
        u (torch.Tensor): Input (b, l, d).
        dt (torch.Tensor): Step size (b, l, d), already positive.
        A (torch.Tensor): State matrix (d, n), negative.
        B (torch.Tensor): Input matrix (b, l, n).
        C (torch.Tensor): Output matrix (b, l, n).
        D (torch.Tensor): Skip weights (d).

    Returns:
        torch.Tensor: Output (b, l, d).
    '''
    u, dt, B, C = u.float(), dt.float(), B.float(), C.float()
    b, l, d = u.shape
    dA = torch.exp(dt.unsqueeze(-1) * A)  # (b, l, d, n)
    dBu = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (b, l, d, n)
    h = u.new_zeros(b, d, A.shape[1])
    ys = []

    for t in range(l):
        h = dA[:, t] * h + dBu[:, t]
        ys.append((h * C[:, t].unsqueeze(1)).sum(-1))

    return torch.stack(ys, 1) + u * D


class MambaBlock(nn.Module):
    '''Causal Mamba block (selective SSM).

    Inputs:
        x (torch.Tensor): (size_batch, len_seq, d_model).
    Outputs:
        torch.Tensor: (size_batch, len_seq, d_model).
    '''

    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ) -> None:
        super().__init__()

        d_inner = expand * d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(
            d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner
        )
        self.x_proj = nn.Linear(
            d_inner, self.dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # dt starts between 0.001 and 0.1 (as in the Mamba paper)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = x.shape[1]
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :l].transpose(1, 2)  # causal
        x = F.silu(x)

        dt, B, C = self.x_proj(x).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt).float())
        A = -torch.exp(self.A_log.float())
        y = selective_scan(x, dt, A, B, C, self.D.float())

        return self.out_proj(y.to(z.dtype) * F.silu(z))


class MambaLSTM(nn.Module):
    '''Decoder: linear -> BiMamba (residual) -> Bi-LSTM -> classifier.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, len_seq, size_in).
    Outputs:
        torch.Tensor: Probabilities (size_batch, len_seq, num_cls).
    '''

    def __init__(
        self,
        size_in: int,
        num_cls: int,
        d_model: int = 64,
        hidden_size: int = 64,
        r_drop: float = 0.2,
    ) -> None:
        super().__init__()

        self.proj = nn.Linear(size_in, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.fwd = MambaBlock(d_model)
        self.bwd = MambaBlock(d_model)
        self.drop = nn.Dropout(r_drop)
        self.lstm = nn.LSTM(
            d_model, hidden_size, 1, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_cls)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        y = self.norm(x)
        y = self.fwd(y) + self.bwd(y.flip(1)).flip(1)  # both directions
        x = x + self.drop(y)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x
