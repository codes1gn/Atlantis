import torch
import time
import torch.profiler as profiler
 
from transformers import YosoConfig, YosoModel, AutoTokenizer, YosoForSequenceClassification

model = YosoForSequenceClassification(YosoConfig(max_position_embeddings=100*512))

# print(model)
batch = 25 * 512

tokenizer = AutoTokenizer.from_pretrained("uw-madison/yoso-4096")
inputs = tokenizer("H"*batch, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
num_labels = len(model.config.id2label)

# model = YosoForSequenceClassification(YosoConfig(num_labels = num_labels))

print(model)
from revlib.utils import module_list_to_momentum_net
# model.yoso.encoder.layer = module_list_to_momentum_net(model.yoso.encoder.layer, residual=True, beta=0.5)  # The only difference

model = model.to("cuda")
inputs = inputs.to("cuda")

labels = torch.tensor([1]).to("cuda")
# loss = model(**inputs, labels=labels).loss

tt=[]
for i in range(1):
    with profiler.profile(activities=[ 
        torch.profiler.ProfilerActivity.CPU, 
        torch.profiler.ProfilerActivity.CUDA,],profile_memory=True,) as prof:
<<<<<<< Updated upstream
        for i in range(3):
            model.train(True)
            model.gradient_checkpointing_enable()
            # with autocast():
            out = model(**inputs,labels=labels)
            # loss = torch.nn.functional.cross_entropy(out, labels)
            loss = out.loss
            # loss.requires_grad_(True)
            loss.backward()
            # scaler.scale(loss).backward()
=======
>>>>>>> Stashed changes
        for i in range(33):
            start_time = time.perf_counter()
            model.train(True)
            model.gradient_checkpointing_enable()
            # with autocast():
            out = model(**inputs,labels=labels)
            # loss = torch.nn.functional.cross_entropy(out, labels)
            loss = out.loss
            # loss.requires_grad_(True)
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
