from torch.utils.checkpoint import checkpoint
from reversyn import Compiler
import torch.profiler as profiler
from transformers.models.swin import SwinForImageClassification, SwinConfig
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights, _swin_transformer
import time
import torch
from benchmarks.models.local import rev_swin
from memory_profiler import memory_usage
import os
import psutil
import gc
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

batch = 80
# print(model)
# tokens = torch.zeros((4, 2048), dtype=torch.long).to("cpu")
tokens = torch.rand((batch, 3, 224, 224), dtype=torch.float32).to("cuda")
labels = torch.rand(batch).to("cuda")

model = SwinForImageClassification(SwinConfig())
# model = swin_t(Swin_T_Weights.DEFAULT)


model = Compiler(model, example_inputs=tokens).reversify()

# model = rev_swin.ReversibleSwinTransformer(drop_path_rate=0.1)
model = model.to("cuda")

ll = []
tt = []
for i in range(1):
    with profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,], profile_memory=True,) as prof:
        for i in range(3):
            start_time = time.perf_counter()
            model.train(True)
            # model.gradient_checkpointing_enable()
            out = model(tokens)
            # out = model(tokens, labels = labels)
            # out = checkpoint(model,tokens).logits
            loss = torch.nn.functional.cross_entropy(out, labels.long())
            # loss = out.loss
            loss.requires_grad_(True)
            t3 = time.perf_counter()
            loss.backward()
            t4 = time.perf_counter()
            print(t4-t3)
            tt.append(time.perf_counter() - start_time)
    print(prof.key_averages().total_average().self_cuda_time_total_str)
    print("cpu:", prof.key_averages().total_average().self_cpu_time_total_str)
    print("cuda:", prof.key_averages().total_average().self_cuda_time_total_str)
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    ll.append(prof.key_averages().total_average().self_cuda_memory_usage)
print(sum(tt)/len(tt))
print(sum(ll)/len(ll))

print("##Checkpoint##")
