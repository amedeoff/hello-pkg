from __future__ import annotations

import math

import torch
from torch import nn


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dimension: int, scale: float = 8.0):
        super().__init__()
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(scale), dimension // 2))
        self.register_buffer("frequencies", frequencies)
        self.projection = nn.Sequential(nn.Linear(dimension, dimension), nn.SiLU())

    def forward(self, time):
        phase = 2 * math.pi * time[:, None] * self.frequencies[None]
        return self.projection(torch.cat([phase.sin(), phase.cos()], dim=-1))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.conv = nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation)
        self.time = nn.Linear(time_dim, channels)

    def forward(self, x, time_embedding):
        update = self.conv(torch.nn.functional.silu(self.norm(x)))
        update = update + self.time(time_embedding)[:, :, None]
        return x + torch.nn.functional.silu(update)


class PromoterCNN(nn.Module):
    def __init__(self, state_channels: int, channels: int = 64, layers: int = 6,
                 center_output: bool = False):
        super().__init__()
        if channels % 8:
            raise ValueError("channels must be divisible by 8")
        self.time_embedding = FourierTimeEmbedding(channels)
        self.input = nn.Conv1d(state_channels + 1, channels, 7, padding=3)
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, channels, 2 ** (index % 4)) for index in range(layers)
        ])
        self.output = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv1d(channels, 4, 1)
        )
        self.center_output = center_output

    def forward(self, state, signal, time):
        values = torch.cat([state.float(), signal.float()], dim=-1).transpose(1, 2)
        hidden = self.input(values)
        embedded_time = self.time_embedding(time.float())
        for block in self.blocks:
            hidden = block(hidden, embedded_time)
        output = self.output(hidden).transpose(1, 2)
        if self.center_output:
            output = output - output.mean(dim=-1, keepdim=True)
        return output


class DDSMScoreAdapter(nn.Module):
    def __init__(self, model: PromoterCNN, max_time: float = 4.0):
        super().__init__()
        self.model = model
        self.max_time = max_time

    def forward(self, combined, time):
        return self.model(combined[..., :4], combined[..., 4:5], time / self.max_time)
