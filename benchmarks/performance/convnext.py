import time
import torch
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

from transformers import ConvNextV2ForImageClassification, ConvNextV2Config, ConvNextV2PreTrainedModel,AutoImageProcessor
# from transformers import AutoModelForImageClassification

tokens = torch.rand((1, 3, 224, 224), dtype=torch.float32).to("cuda")
tokens =tokens.requires_grad_()
labels = torch.rand(1).to("cuda")

from datasets import load_dataset
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]



image_processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224")
model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224")

# import timm
# model = timm.create_model(model_name='convnextv2_nano', pretrained=True)

inputs = image_processor(image, return_tensors="pt")
inputs = inputs.to("cuda")
# model = ConvNextV2ForImageClassification(ConvNextV2Config())
model = model.to("cuda")
print(model)


from revlib.utils import module_list_to_momentum_net, sequential_to_momentum_net
model.convnextv2.encoder.stages = module_list_to_momentum_net(model.convnextv2.encoder.stages, residual=True, beta=0.5)  # The only difference

# model = torch.load("/data/workspace/reverse/reversify/benchmarks/models/local/revcol.pth")

ll = [] 
tt=[]
for i in range(3):
    with profiler.profile(activities=[ 
        torch.profiler.ProfilerActivity.CPU, 
        torch.profiler.ProfilerActivity.CUDA,],profile_memory=True,) as prof:
        for i in range(3):
            start_time = time.perf_counter()
            model.train(True)
            # with autocast():
            out = model(**inputs).logits
            # out =checkpoint( model,tokens).logits
            loss = torch.nn.functional.cross_entropy(out, labels.long())
            loss.requires_grad_(True)
            loss.backward()
            tt.append(time.perf_counter() - start_time)
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    print("cpu:", prof.key_averages().total_average().self_cpu_time_total_str)
    print("cuda:", prof.key_averages().total_average().self_cuda_time_total_str)
    ll.append(prof.key_averages().total_average().self_cuda_memory_usage)
print(sum(tt)/len(tt))
print(sum(ll)/len(ll))
