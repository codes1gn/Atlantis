from reversyn import Compiler
from reversyn import MetaRunner
import copy
import gc
import logging
import os
import random
import time

import numpy as np
import psutil
import torch
import torch.profiler as profiler
from torch.cuda.amp import autocast as autocast
from torch.utils.checkpoint import checkpoint

from benchmarks.models.local import resnet

batch = 16
tokens = torch.rand((batch, 3, 224, 224), requires_grad=True,
                    dtype=torch.float32).to("cuda")
labels = torch.rand(batch).to("cuda")

model = resnet.resnet110

with torch.no_grad():
    compiler = Compiler(model, example_inputs=torch.rand(
        1, 3, 224, 224), device="cuda")
model = compiler.reversify()
print('compile time: ', compiler.compile_time)

model = model.to("cuda")
torch.cuda.empty_cache()

# ll = []
tt = []
for i in range(1):
    with profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,], profile_memory=False,) as prof:
        for i in range(3):
            start_time = time.perf_counter()
            model.train(True)
            if (True):
                # with autocast():
                # with torch.autograd.graph.save_on_cpu(pin_memory=True) :
                # out = model(tokens)
                out = checkpoint(model, tokens)
                loss = torch.nn.functional.cross_entropy(out, labels.long())
                loss.requires_grad_(True)
                t3 = time.perf_counter()
                loss.backward()
            t4 = time.perf_counter()
            print(t4-t3)
            tt.append(time.perf_counter() - start_time)
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
    print("cpu:", prof.key_averages().total_average().self_cpu_time_total_str)
    print("cuda:", prof.key_averages().total_average().self_cuda_time_total_str)
    # ll.append(prof.key_averages().total_average().self_cuda_memory_usage)
print(sum(tt[1:])/len(tt[1:]))
# print(sum(ll[1:])/len(ll[1:]))
