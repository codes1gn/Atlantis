# https://github.com/lucidrains/reformer-pytorch/
import torch
from torch import nn
from reformer_pytorch.reformer_pytorch import LSHSelfAttention, Chunk, FeedForward, AbsolutePositionalEmbedding
import revlib


class Reformer(torch.nn.Module):
    def __init__(self, sequence_length: int, features: int, depth: int, heads: int, bucket_size: int = 64,
                 lsh_hash_count: int = 8, ff_chunks: int = 16, input_classes: int = 256, output_classes: int = 256):
        super(Reformer, self).__init__()
        self.token_embd = nn.Embedding(input_classes, features * 2)
        self.pos_embd = AbsolutePositionalEmbedding(features * 2, sequence_length)

        self.core = revlib.ReversibleSequential(*[nn.Sequential(nn.LayerNorm(features), layer) for _ in range(depth)
                                                  for layer in
                                                  [LSHSelfAttention(features, heads, bucket_size, lsh_hash_count),
                                                   Chunk(ff_chunks, FeedForward(features, activation=nn.GELU),
                                                         along_dim=-2)]],
                                                split_dim=-1)
        self.out_norm = nn.LayerNorm(features * 2)
        self.out_linear = nn.Linear(features * 2, output_classes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.out_linear(self.out_norm(self.core(self.token_embd(inp) + self.pos_embd(inp))))


sequence = 1024
classes = 16
model = Reformer(sequence, 256, 6, 8, output_classes=classes)
out = model(torch.ones((16, sequence), dtype=torch.long))
assert out.size() == (16, sequence, classes)