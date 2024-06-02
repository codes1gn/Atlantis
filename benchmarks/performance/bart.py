from reversyn import Compiler
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
import torch.profiler as profiler
import time
import torch
from torch.utils.checkpoint import checkpoint
from transformers import BartForSequenceClassification, AutoTokenizer, BartConfig

# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
model = BartForSequenceClassification(BartConfig())

batch = 10 * 512
tokenizer = AutoTokenizer.from_pretrained("valhalla/bart-large-sst2")
# model = BartForSequenceClassification.from_pretrained("valhalla/bart-large-sst2")
model = BartForSequenceClassification(
    BartConfig(max_position_embeddings=100*512))
model = model.to("cpu")

inputs = tokenizer("h"*batch, return_tensors="pt")
inputs = inputs.to("cpu")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]
num_labels = len(model.config.id2label)
# model = BartForSequenceClassification.from_pretrained("valhalla/bart-large-sst2", num_labels=num_labels)
model = BartForSequenceClassification(BartConfig(
    max_position_embeddings=100*512, num_labels=num_labels))

inputs = inputs.to("cuda")
labels = torch.tensor([1]).to("cuda")
model = model.to("cuda")

scaler = GradScaler()


model = Compiler(model, example_inputs=inputs).reversify()

ll = []
tt = []
for i in range(3):
    with profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,], profile_memory=True,) as prof:
        for i in range(3):
            start_time = time.perf_counter()
            model.train(True)
            # with autocast():
            model.gradient_checkpointing_enable()
            out = model(**inputs, labels=labels)
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
