from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import datasets, linear_model
from reversyn import MetaRunner
from revlib.utils import module_list_to_momentum_net, sequential_to_momentum_net
import copy
import gc
import logging
import os
import random
import time

import numpy as np
import psutil
import torch
import torch.profiler as profiler
from iunets import iUNet
from iunets.baseline_networks import StandardUNet
from torch.cuda.amp import autocast as autocast
from torch.utils.checkpoint import checkpoint
from torchvision.models.vision_transformer import _vision_transformer
from transformers import (
    AutoFeatureExtractor,
    AutoModelForSequenceClassification,
    BertConfig,
    BertForSequenceClassification,
    DeiTForImageClassification,
)

from benchmarks.models.local import resnet

batch = 10
# tokens = torch.rand((batch, 3, 224, 224), requires_grad=True, dtype=torch.float32)
labels = torch.rand(batch).to("cuda")

baseline = resnet.resnet101_1k

# baseline = StandardUNet(3, dim=2)
# baseline = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224')
# baseline = BertForSequenceClassification(BertConfig(num_labels=1000))

model = copy.deepcopy(baseline)

# revnet = resnet.revnet38_1k


# model.encoder.layers = module_list_to_momentum_net(model.encoder.layers, residual=False, beta=0.5)  # The only difference
# model.deit.encoder.layer = module_list_to_momentum_net(model.deit.encoder.layer, residual=False, beta=0.5)  # The only difference
# model.bert.encoder.layer = module_list_to_momentum_net(model.bert.encoder.layer, residual=True, beta=0.5)  # The only difference

with torch.no_grad():
    runner = MetaRunner(
        model, torch.rand((1, 3, 224, 224), dtype=torch.float32), device="cpu"
    )
model = runner.run()
print("compile:", runner.end_time - runner.start_time)


# revnet = revnet.to("cuda")
# baseline = baseline.to("cuda")
# rev_model = rev_model.to("cuda")

torch.cuda.empty_cache()

# target = np.array([])
# out1 = np.array([])
# out2 = np.array([])

target = []
# out1 = []
out2 = []
for i in range(1):
    tokens = torch.rand((batch, 3, 224, 224), dtype=torch.float32)
    input_ids = torch.ones((batch, 512), dtype=torch.long)
    attention_mask = torch.ones((batch, 512), dtype=torch.long)
    labels = torch.ones((batch), dtype=torch.long)
    # np.append(target, baseline(tokens))
    # np.append(out1, revnet(tokens))
    # np.append(out2, rev_model(tokens))
    target.append(baseline(tokens).detach().numpy())
    # target.append(baseline(input_ids, attention_mask=attention_mask, labels=labels).logits.detach().numpy())
    # out1.append(revnet(tokens))
    out2.append(model(tokens).detach().numpy())
    # out2.append(model(input_ids, attention_mask=attention_mask, labels=labels).logits.detach().numpy())

target = np.array(target)
# out1=np.array(out1)
out2 = np.array(out2)
print(target.shape)
# print(target)
# print(out1)
# print(out2)

print(target.max())
print(out2.max())

print(target.mean())
print(out2.mean())

np.save("base.npy", target)
# np.save("target.npy", target)
np.save("rev.npy", out2)


real_y_true_mask = 1 - (target == 0)

mask = real_y_true_mask
y_label = np.array(target)
y_predict = np.array(out2)
val = ((y_label - y_predict) ** 2) * real_y_true_mask
mse = np.sum(val) / real_y_true_mask.sum()
print(mse)
rmse = np.sqrt(mse)
print(rmse)


def MAPE1(y_true, y_pred, mask):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    loss = np.abs((y_true - y_pred) / y_true)
    loss *= mask
    non_zero_len = mask.sum()
    np.mean() * 100
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def MAPE(labels, predicts, mask):
    """
    Mean absolute percentage. Assumes ``y >= 0``.
    Defined as ``(y - y_pred).abs() / y.abs()``
    """
    loss = np.abs(predicts - labels) / (np.abs(labels) + 1)
    loss *= mask
    non_zero_len = mask.sum()
    return np.sum(loss) / non_zero_len


def mae(a, b):
    mask = a != 0
    return (np.fabs(a - b) / a)[mask].mean()


print(MAPE(target, out2, real_y_true_mask))
# from sklearn.metrics import mean_absolute_percentage_error

# print(mean_absolute_percentage_error(np.squeeze(target), np.squeeze(out2)))
