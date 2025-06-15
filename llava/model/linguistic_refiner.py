import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


class AdaptiveGate(nn.Module):
    """Adaptive gating module from LTR paper"""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, pooled, timestep_emb=None):
        if timestep_emb is not None:
            gated = x + timestep_emb
        else:
            gated = x
        gated = gated + pooled
        gate = torch.sigmoid(self.linear(gated))
        return x * gate


class LinguisticRefiner(nn.Module):
    """A light Transformer stack with full attention and adaptive gating."""

    def __init__(self, hidden_size, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([LlamaDecoderLayer(hidden_size) for _ in range(num_layers)])
        self.gates = nn.ModuleList([AdaptiveGate(hidden_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, timestep_emb=None):
        pooled = hidden_states.mean(dim=1, keepdim=True)
        for layer, gate in zip(self.layers, self.gates):
            hidden_states = layer(hidden_states)[0]
            hidden_states = gate(hidden_states, pooled, timestep_emb)
        hidden_states = self.norm(hidden_states)
        return hidden_states
