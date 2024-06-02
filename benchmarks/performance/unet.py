import time
from iunets.baseline_networks import StandardUNet
from iunets import iUNet
import torch
import torch.profiler as profiler
from torch.utils.checkpoint import checkpoint

batch = 30
tokens = torch.rand((batch, 3, 224, 224), requires_grad=True,
                    dtype=torch.float32).to("cuda")
labels = torch.rand(batch, 224, 224).to("cuda")

model = StandardUNet(3, dim=2)

model = iUNet(
    3,
    channels=(7, 15, 35, 91),
    dim=2,
    architecture=(2, 3, 1, 3),
)

# from reversyn import Compiler

# model = Compiler(model, example_inputs=tokens).reversify()

# from reversyn import MetaRunner
# runner = MetaRunner(model, torch.rand((1, 3, 224, 224), dtype=torch.float32), device="cuda")
# model1 = runner.run().to("cuda")
# print("compile:", runner.end_time - runner.start_time)
# del model
# torch.cuda.empty_cache()
# model = model1
# del model1
# torch.cuda.empty_cache()

model = model.to("cuda")

# ll = []
tt = []
for i in range(3):
    with profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,], profile_memory=False,) as prof:
        for i in range(3):
            start_time = time.perf_counter()
            model.train(True)
            # with autocast():
            # out = model(tokens)
            out = checkpoint(model, tokens)
            loss = torch.nn.functional.cross_entropy(out, labels.long())
            loss.requires_grad_(True)
            t3 = time.perf_counter()
            loss.backward()
            t4 = time.perf_counter()
            print(t4-t3)
            tt.append(time.perf_counter() - start_time)
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
    print("cpu:", prof.key_averages().total_average().self_cpu_time_total_str)
    print("cuda:", prof.key_averages().total_average().self_cuda_time_total_str)
    # ll.append(prof.key_averages().total_average().self_cuda_memory_usage)
print(sum(tt[3:])/len(tt[3:]))
# print(sum(ll[1:])/len(ll[1:]))
