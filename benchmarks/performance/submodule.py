import time

import memcnn
import revlib
import torch
from torch import nn

channels = 64
depth = 16
momentum_ema_beta = 0.1


class ExampleOperation(nn.Module):
    def __init__(self, channels):
        super(ExampleOperation, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.Identity(),
            # nn.BatchNorm2d(num_features=channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)


# Compute y2 from x2 and f(x1) by merging x2 and f(x1) in the forward pass.
def momentum_coupling_forward(
    other_stream: torch.Tensor, fn_out: torch.Tensor
) -> torch.Tensor:
    return other_stream * momentum_ema_beta + fn_out * (1 - momentum_ema_beta)


# Calculate x2 from y2 and f(x1) by manually computing the inverse of momentum_coupling_forward.
def momentum_coupling_inverse(
    output: torch.Tensor, fn_out: torch.Tensor
) -> torch.Tensor:
    return (output - fn_out * (1 - momentum_ema_beta)) / momentum_ema_beta


additive_coupling_inverse = revlib.additive_coupling_inverse
additive_coupling_forward = revlib.additive_coupling_forward


# Compute y2 from x2 and f(x1) by merging x2 and f(x1) in the forward pass.
def affine_coupling_forward(
    other_stream: torch.Tensor, fn_out: torch.Tensor
) -> torch.Tensor:
    return other_stream * momentum_ema_beta + fn_out * (1 - momentum_ema_beta)


# Calculate x2 from y2 and f(x1) by manually computing the inverse of momentum_coupling_forward.
def affine_coupling_inverse(output: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return (output - fn_out * (1 - momentum_ema_beta)) / momentum_ema_beta


# Pass in coupling functions which will be used instead of x2 + f(x1) and y2 - f(x1)
# rev_model = revlib.ReversibleSequential(*[layer for _ in range(depth)
#                                           for layer in [nn.Conv2d(channels, channels, (3, 3), padding=1),
#                                                         nn.Identity()]],
#                                         coupling_forward=[momentum_coupling_forward, additive_coupling_forward],
#                                         coupling_inverse=[momentum_coupling_inverse, additive_coupling_inverse])

rev_model = revlib.ReversibleModule(
    ExampleOperation,
    coupling_forward=[momentum_coupling_forward],
    coupling_inverse=[momentum_coupling_inverse],
)

invertible_module = memcnn.AffineCoupling(
    Fm=ExampleOperation(channels), Gm=ExampleOperation(channels), adapter=memcnn.AffineAdapterNaive
)
invertible_module_wrapper = memcnn.InvertibleModuleWrapper(
    fn=invertible_module, keep_input=False, keep_input_inverse=False
)

inp = torch.randn((16, channels * 2, 224, 224))

labels = torch.randn((16, channels * 2, 224, 224))

model = rev_model
model = invertible_module_wrapper
model = ExampleOperation(channels*2)
model = model.to("cuda")
inp = inp.to("cuda")
labels = labels.to("cuda")

# ll = []
tt = []
for i in range(10):
    for i in range(3):
        start_time = time.perf_counter()
        model.train(True)
        if True:
            # with autocast():
            # with torch.autograd.graph.save_on_cpu(pin_memory=True) :
            out = model(inp)
            loss = torch.nn.functional.cross_entropy(out, labels)
            loss.requires_grad_(True)
            t3 = time.perf_counter()
            loss.backward()
        # t4 = time.perf_counter()
        # print(t4-t3)
        tt.append(time.perf_counter() - start_time)
print(sum(tt[1:]) / len(tt[1:]))
# print(sum(ll[1:])/len(ll[1:]))
