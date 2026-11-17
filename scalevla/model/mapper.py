"""Feature mapper: backbone tokens -> the conditioning width of the action head (2048).

forward(features [B, T, d_in], token_type [B, T] int8 (0 text, 1 + c image of camera c, -1 padding))
  -> {"backbone_features": [B, T, d_out], "backbone_attention_mask": [B, T] bool, "image_mask": [B, T] bool}
LayerNorm(d_in) -> Linear(d_in, rank, no bias) -> Linear(rank, d_out); every token is kept with its label. The action head
then applies its feature adapter (LayerNorm and self-attention layers), and its DiT attends to image and non-image tokens
in turn through the two masks.
"""
from torch import nn

MAPPER_KIND = "per_token"  # recorded in policy.json


class FeatureMapper(nn.Module):
    def __init__(self, d_in, d_out=2048, rank=2048):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.down = nn.Linear(d_in, rank, bias=False)
        self.up = nn.Linear(rank, d_out)

    def forward(self, features, token_type):
        return {"backbone_features": self.up(self.down(self.norm(features))),
                "backbone_attention_mask": token_type >= 0, "image_mask": token_type > 0}
