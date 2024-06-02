from reversyn import Compiler
from torch.utils.checkpoint import checkpoint
import gc
import os
import time

import numpy as np
import psutil
import torch
import torch.profiler as profiler
from memory_profiler import memory_usage
from reformer_pytorch import ReformerLM
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from transformers import (
    AutoModelForCausalLM,
    GPT2Tokenizer,
    GPTNeoConfig,
    GPTNeoForCausalLM,
)

scaler = GradScaler()

# base max batch
# batch = 40

batch = 2

# tokens = torch.zeros((4, 2048), dtype=torch.long).to("cpu")
tokens = torch.zeros((batch, 100), dtype=torch.long).to("cuda")
# labels = torch.zeros((batch,50257), dtype=torch.long).to("cuda")
# input_ids = torch.zeros((8, 512), dtype=torch.long).to("cuda")
attention_mask = torch.zeros((batch, 100), dtype=torch.long).to("cuda")
# labels = torch.zeros((397), dtype=torch.long).to("cuda")
labels = torch.zeros((batch, 100), dtype=torch.long).to("cuda")


# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
model = GPTNeoForCausalLM(
    GPTNeoConfig(
        attention_dropout=0,
        attention_layers=[
            "global",
            "local",
            "global",
            "local",
            "global",
            "local",
            "global",
            "local",
            "global",
            "local",
            "global",
            "local",
        ],
        attention_types=[[["global", "local"], 6]],
        bos_token_id=50256,
        eos_token_id=50256,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        vocab_size=50257,
        window_size=256,
    )
)

# model = ReformerLM(
#     num_tokens= 50257,
#     dim = 1024,
#     depth = 12,
#     max_seq_len = 4096,
#     lsh_dropout = 0.1,
#     causal = True,
#     full_attn_thres = 1024
# )


model = Compiler(model, example_inputs=tokens).reversify()

# tt = []
# ll = []
# for i in range(1):
#     with profiler.profile(
#         activities=[
#             torch.profiler.ProfilerActivity.CPU,
#             torch.profiler.ProfilerActivity.CUDA,
#         ],
#         profile_memory=True,
#     ) as prof:
#         for i in range(3):
#             start_time = time.perf_counter()
#             model.train(True)
#             # model.gradient_checkpointing_enable()
#             if True:
#                 # with autocast():
#                 # with torch.autograd.graph.save_on_cpu(pin_memory=True) and autocast():
#                 # out = model(tokens)
#                 out = model(tokens, attention_mask=attention_mask, labels=labels)
#                 # out = checkpoint(model,tokens, attention_mask, labels)
#                 # loss = torch.nn.functional.cross_entropy(out, labels)
#                 loss = out.loss
#                 loss.requires_grad_(True)
#                 # with torch.autograd.graph.save_on_cpu(pin_memory=True):
#                 t3 = time.perf_counter()
#                 loss.backward()
#                 t4 = time.perf_counter()
#                 print(t4 - t3)
#             # scaler.scale(loss).backward()
#             tt.append(time.perf_counter() - start_time)
#     print("cpu:", prof.key_averages().total_average().self_cpu_time_total_str)
#     print("cuda:", prof.key_averages().total_average().self_cuda_time_total_str)
#     ll.append(prof.key_averages().total_average().self_cuda_memory_usage)
# print(sum(tt) / len(tt))
# print(sum(ll) / len(ll))
