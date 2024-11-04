import time
import torch
import torch.profiler as profiler

from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("vinai/bertweet-base")

# For transformers v4.x+:
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

# For transformers v3.x:
# tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

# INPUT TWEET IS ALREADY NORMALIZED!
line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"

# input_ids = torch.tensor([tokenizer.encode(line)])
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    features = model(**inputs)  # Models outputs are now tuples

# With TensorFlow 2.0+:
# from transformers import TFAutoModel
# bertweet = TFAutoModel.from_pretrained("vinai/bertweet-base")
    

labels = torch.tensor([1])

print(model)
from revlib.utils import module_list_to_momentum_net
model.encoder.layer = module_list_to_momentum_net(model.encoder.layer, residual=True, beta=0.5)  # The only difference

inputs = inputs.to("cuda")
model = model.to("cuda")
labels = labels.to("cuda")

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
            # model.gradient_checkpointing_enable() 
            out = model(**inputs)
            # print(out[1])
            # out = checkpoint(model, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # out = checkpoint(model, tokens['input_ids']).logits
            loss = torch.nn.functional.cross_entropy(out[1], labels.long())
            # loss.requires_grad_(True)
            # loss = out.loss
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
