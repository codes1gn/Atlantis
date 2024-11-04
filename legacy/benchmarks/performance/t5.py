import time
import torch
from torch.utils.checkpoint import checkpoint
import torch.profiler as profiler
from torch.cuda.amp import autocast as autocast

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5ForSequenceClassification
tokenizer = T5Tokenizer.from_pretrained("t5-small")
input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids

model = T5ForSequenceClassification(T5Config(
    decoder_start_token_id=0
))
from revlib.utils import module_list_to_momentum_net
# model.encoder.block = module_list_to_momentum_net(model.encoder.block, residual=True, beta=0.5)  # The only difference
# model.decoder.block = module_list_to_momentum_net(model.decoder.block, residual=True, beta=0.5)  # The only difference
# model.decoder.block[0].layer = module_list_to_momentum_net(model.decoder.block[0].layer, residual=True, beta=0.5)  # The only difference

print(model)


tt=[]
for i in range(3):
    with profiler.profile(activities=[ 
        torch.profiler.ProfilerActivity.CPU, 
        torch.profiler.ProfilerActivity.CUDA,],profile_memory=True,) as prof:
        for i in range(3):
            start_time = time.perf_counter()
            model.train(True)
            # model.gradient_checkpointing_enable()
            # with autocast():
            out = model(input_ids=input_ids, labels=labels)
            # out = checkpoint(model,tokens, attention_mask, labels)
            # loss = torch.nn.functional.cross_entropy(out, labels)
            loss = out.loss
            loss.requires_grad_(True)
            print(loss)
            t3 = time.perf_counter()
            loss.backward()
            t4 = time.perf_counter()
            print(t4-t3)
            # scaler.scale(loss).backward()
            tt.append(time.perf_counter() - start_time)
    print("cpu:", prof.key_averages().total_average().self_cpu_time_total_str)
    print("cuda:", prof.key_averages().total_average().self_cuda_time_total_str)
print(sum(tt)/len(tt))
