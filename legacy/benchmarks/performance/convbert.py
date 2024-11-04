from revlib.utils import module_list_to_momentum_net
import time
import torch
from transformers import ConvBertForSequenceClassification, ConvBertConfig, AutoTokenizer, ConvBertTokenizer
import torch.profiler as profiler


tokenizer = AutoTokenizer.from_pretrained("YituTech/conv-bert-base")
# tokenizer = ConvBertTokenizer
# model = ConvBertForSequenceClassification.from_pretrained("YituTech/conv-bert-base")
model = ConvBertForSequenceClassification(ConvBertConfig())

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()

# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
# model = ConvBertForSequenceClassification.from_pretrained("YituTech/conv-bert-base", num_labels=num_labels)
model = ConvBertForSequenceClassification(
    ConvBertConfig(num_labels=num_labels))

labels = torch.tensor([1])
loss = model(**inputs, labels=labels).loss

print(model)
# model.convbert.encoder.layer = module_list_to_momentum_net(model.convbert.encoder.layer, residual=True, beta=0.5)  # The only difference

inputs = inputs.to("cuda")
model = model.to("cuda")
labels = labels.to("cuda")

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
