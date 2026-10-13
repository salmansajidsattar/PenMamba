"""E3: first encoder layer split by sensor group, mixed by a gate.

Replaces the baseline PatchEmbed (13 -> 64 channels, kernel 2, stride 2).
Everything after it is the same as BLConv.
"""
import torch
import torch.nn as nn

from .conv import BLConv

__all__ = ['GroupBLConv']

# (start, end) channel ranges: front acc, rear acc, gyro, mag, force
GROUPS = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 13)]


class Expert(nn.Module):
    '''Small stem for one sensor group: conv -> norm -> GELU -> depthwise conv.'''

    def __init__(self, in_chan: int, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_chan, dim, 2, 2)
        self.norm = nn.InstanceNorm1d(dim)
        self.act = nn.GELU()
        self.dw = nn.Conv1d(dim, dim, 5, padding='same', groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(self.act(self.norm(self.conv(x))))


class GroupEmbed(nn.Module):
    '''Five sensor-group experts, mixed with softmax weights from a gate.

    Inputs:
        x (torch.Tensor): (size_batch, 13, len_seq).
    Outputs:
        torch.Tensor: (size_batch, dim, len_seq // 2).
    '''

    def __init__(self, in_chan: int, dim: int, learned: bool = True) -> None:
        super().__init__()
        assert in_chan == GROUPS[-1][1], 'Groups are defined for 13 channels.'
        self.experts = nn.ModuleList(
            [Expert(b - a, dim) for a, b in GROUPS]
        )
        self.learned = learned
        if learned:
            self.gate = nn.Conv1d(in_chan, len(GROUPS), 2, 2)
            nn.init.zeros_(self.gate.weight)  # start with equal weights
            nn.init.zeros_(self.gate.bias)
        self.norm = nn.InstanceNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = torch.stack(
            [e(x[:, a:b]) for e, (a, b) in zip(self.experts, GROUPS)], dim=1
        )  # (B, 5, dim, T')
        if self.learned:
            w = torch.softmax(self.gate(x), dim=1)  # (B, 5, T')
            y = (outs * w.unsqueeze(2)).sum(dim=1)
        else:
            y = outs.mean(dim=1)  # control: equal weights
        return self.norm(y)


class GroupBLConv(BLConv):
    '''BLConv whose first layer is GroupEmbed.'''

    def __init__(
        self,
        in_chan: int,
        depths: list[int] = [1, 1, 1],
        dims: list[int] = [64, 128, 256],
        learned: bool = True,
    ) -> None:
        super().__init__(in_chan, depths, dims)
        self.layers[0] = GroupEmbed(in_chan, dims[0], learned)
