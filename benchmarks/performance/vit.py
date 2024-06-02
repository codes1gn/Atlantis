import time
import torch
from benchmarks.models.local import rev_vit
from memory_profiler import memory_usage
import os, psutil
import gc
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
from torchvision.models import vit_b_32,ViT_B_32_Weights
from torchvision.models.vision_transformer import _vision_transformer
import torch.profiler as profiler
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
scaler = GradScaler()
from torch.utils.checkpoint import checkpoint

batch = 350
tokens = torch.rand((batch, 3, 224, 224), requires_grad=True, dtype=torch.float32).to("cuda")
# tokens =tokens.requires_grad_()
labels = torch.rand(batch).to("cuda")

# model = vit_b_32(ViT_B_32_Weights.DEFAULT)
model = _vision_transformer(
        image_size=224,
        patch_size=32,
        num_layers=8,
        num_heads=8,
        hidden_dim=768,
        mlp_dim=3072,
        weights=None,
        # weights=ViT_B_32_Weights.DEFAULT,
        progress=True
    )

# model = rev_vit.RevViT(patch_size=(32,32), image_size=(224,224),num_classes=1000)
model = model.to("cuda")


from revlib.utils import module_list_to_momentum_net, sequential_to_momentum_net
model.encoder.layers = module_list_to_momentum_net(model.encoder.layers, residual=False, beta=0.5)  # The only difference


tt=[]
ll=[]
for i in range(1):
   
        for i in range(33):
            start_time = time.perf_counter()
            model.train(True)
            # with autocast():
            # out = model(tokens)
            out =checkpoint( model,tokens)
            loss = torch.nn.functional.cross_entropy(out, labels.long())
            t3 = time.perf_counter()
            with profiler.profile(activities=[ 
            torch.profiler.ProfilerActivity.CPU, 
            torch.profiler.ProfilerActivity.CUDA,],profile_memory=True,) as prof:
                loss.backward()
            t4 = time.perf_counter()
            print(t4-t3)
            tt.append(time.perf_counter() - start_time)
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
            print("cpu:", prof.key_averages().total_average().self_cpu_time_total_str)
            print("cuda:", prof.key_averages().total_average().self_cuda_time_total_str)
            ll.append(prof.key_averages().total_average().self_cuda_memory_usage)
print(sum(tt)/len(tt))
print(sum(ll)/len(ll)/1024**2)

