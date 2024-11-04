import torch.profiler as profiler
import time
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForSequenceClassification, BeitForImageClassification, BeitConfig, BeitForSemanticSegmentation

# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
model = BeitForImageClassification(BeitConfig())

import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
scaler = GradScaler()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from tqdm.auto import tqdm

input_ids = torch.ones((1,3,224,224)).to("cuda")
attention_mask = torch.ones((8, 512)).to("cuda")
labels = torch.ones((1,2)).to("cuda")

print(input_ids.shape)
print(attention_mask.shape)
print(labels.shape)


ll = [] 
tt = []
for i in range(3):
    with profiler.profile(activities=[ 
        torch.profiler.ProfilerActivity.CPU, 
        torch.profiler.ProfilerActivity.CUDA,],profile_memory=True,) as prof:
        for i in range(3):
            start_time = time.perf_counter()
            model.train(True)
            # with autocast():
            model.gradient_checkpointing_enable()
            print(model.supports_gradient_checkpointing)
            out = model(input_ids, labels=labels)
            # out = checkpoint(model, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # out = checkpoint(model, tokens['input_ids']).logits
            # loss = torch.nn.functional.cross_entropy(out, labels.long())
            # loss.requires_grad_(True)
            loss = out.loss
            t3 = time.perf_counter()
            loss.backward()
            t4 = time.perf_counter()
            print(t4-t3)
            tt.append(time.perf_counter() - start_time)
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
    print("cpu:", prof.key_averages().total_average().self_cpu_time_total_str)
    print("cuda:", prof.key_averages().total_average().self_cuda_time_total_str)
    ll.append(prof.key_averages().total_average().self_cuda_memory_usage)
print(sum(tt)/len(tt))
print(sum(ll)/len(ll))
