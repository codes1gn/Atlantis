import torch
from benchmarks.models.local import revnet,resnet
import time

# model = getattr(revnet, "revnet38")()

# print(model)

model = revnet.RevNet(
            units=[9, 9, 9],
            filters=[32, 32, 64, 112],
            strides=[1, 2, 2],
            classes=10
            )
model = resnet.revnet104_1k

batch = 16
tokens = torch.rand((batch, 3, 224, 224), requires_grad=True,dtype=torch.float32).to("cuda")
labels = torch.rand(batch).to("cuda")


model = model.to("cuda")
torch.cuda.empty_cache()

# ll = [] 
tt = []
for i in range(1):
   
        for i in range(3):
            start_time = time.perf_counter()
            model.train(True)
            # if(True):
            # with autocast():
            # with torch.autograd.graph.save_on_cpu(pin_memory=True) :
            out = model(tokens)
            # out = checkpoint(model, tokens)
            loss = torch.nn.functional.cross_entropy(out, labels.long())
            loss.requires_grad_(True)
            t3 = time.perf_counter()
            loss.backward()
            t4 = time.perf_counter()
            print(t4-t3)
            tt.append(time.perf_counter() - start_time)
   
print(sum(tt[1:])/len(tt[1:]))
# print(sum(ll[1:])/len(ll[1:]))

