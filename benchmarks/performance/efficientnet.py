import torch.profiler as profiler
import time
import torch
from torch.utils.checkpoint import checkpoint


efficientnet = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                       'nvidia_convnets_processing_utils')

tokens = torch.rand((1, 3, 224, 224), requires_grad=True,
                    dtype=torch.float32).to("cuda")
labels = torch.rand(1).to("cuda")

model = efficientnet.to("cuda")

# from reversyn import MetaRunner
# runner = MetaRunner(model, tokens, device="cuda")
# model1 = runner.run().to("cuda")
# print("compile:", runner.end_time - runner.start_time)
# del model
# torch.cuda.empty_cache()
# model = model1
# del model1

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
