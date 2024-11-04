from reversyn import Compiler
from transformers import AutoFeatureExtractor, DeiTForImageClassification
from torch.utils.checkpoint import checkpoint
from torchvision.models.vision_transformer import _vision_transformer
from torchvision.models import ViT_B_32_Weights, vit_b_32
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import torch.profiler as profiler
import gc
import logging
import os
import time

import numpy as np
import psutil
import torch
from memory_profiler import memory_usage

from benchmarks.models.local import rev_vit

logging.basicConfig(level=logging.DEBUG)

scaler = GradScaler()

batch = 64
tokens = torch.rand((batch, 3, 224, 224), requires_grad=True, dtype=torch.float32).to(
    "cuda"
)
tokens = tokens.requires_grad_()
labels = torch.rand(batch, 1000).to("cuda")

model = DeiTForImageClassification.from_pretrained(
    "facebook/deit-base-distilled-patch16-224"
)
model = model.to("cuda")

# model = rev_vit.RevViT(image_size=(32,32),num_classes=1000)
# model = model.to("cuda")
# print(model)


model = Compiler(model, example_inputs=tokens).reversify()


tt = []
for i in range(3):
    with profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
    ) as prof:
        for i in range(3):
            start_time = time.perf_counter()
            model.train(True)
            # with autocast():
            # model.gradient_checkpointing_enable()
            out = model(tokens, labels=labels)
            # out =checkpoint( model,tokens)
            # loss = torch.nn.functional.cross_entropy(out, labels.long())
            loss = out.loss
            t3 = time.perf_counter()
            loss.backward()
            t4 = time.perf_counter()
            print(t4 - t3)
            tt.append(time.perf_counter() - start_time)
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    print("cpu:", prof.key_averages().total_average().self_cpu_time_total_str)
    print("cuda:", prof.key_averages().total_average().self_cuda_time_total_str)
print(sum(tt) / len(tt))
