import os
import re
from enum import Enum

import timm
import torch
import torchvision
from benchmarks.models.local import *
from benchmarks.models.timm import *
from benchmarks.models.torchvision import *
from benchmarks.models.transformers import *

# from benchmarks.dataset import get_dataloader, get_dataset, get_tokenized_dataset
from benchmarks.dataset import get_dataloader, get_dataset

# from torchvision import swin_t

# import benchmarks.models as models
# from benchmarks.models.local import resnet as local_resnet

NORMALIZE_DICT = {
    "cifar10": dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    "cifar100": dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    "cifar10_224": dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    "cifar100_224": dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
}


class ModelSource(Enum):
    DEFAULT = "default"
    LOCAL = "local"
    TORCHVISION = "torchvision"
    TRANSFORMERS = "transformers"
    TIMM = "timm"


MODEL_DICT = {
    # revnet paper
    "resnet50": local_resnet.resnet50,
    "revnet50": local_resnet.revnet50,
    "resnet101": local_resnet.resnet101,
    "revnet101": local_resnet.revnet101,
    "resnet152": local_resnet.resnet152,
    "revnet152": local_resnet.revnet152,
    # vit
    # "vit_t": timm_vit_t,
    "vit_b_16": torchvision.models.vit_b_16(),
    "vit_l_16": torchvision.models.vit_l_16(),
    # swin transformer
    "swin_t": swin_t(Swin_T_Weights.DEFAULT),
    # "swin_t": _swin_transformer(num_heads=1000),
    # deit
    # "deit_b": transformers_deit_b,
    "deit_s": timm_deit_s,
    "deit_b": timm_deit_b,
    # bert
    "bert": transformers_bert,
    # gpt
    "gpt_neo": transformers_gpt_neo,
    # others
    # "vit": ViTForImageClassification(ViTConfig(num_labels=10)),
    "vgg": torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True),
    "densenet": torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True),
    "googlenet": torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True),
    "resnext": torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True),
    "efficientnet": timm_efficientnet_b0
}

# TRANSFORMER_MODEL_DICT = {
#     # "resnet": models.transformers.resnet,
#     # "vit": ViTForImageClassification(ViTConfig(num_labels=10)),
# }

TIMM_MODEL_DICT = {
    # "vit": ViTForImageClassification(ViTConfig(num_labels=10)),
}


def get_model_source(model):
    module_name = model.__module__
    if module_name.startswith("transformers.models"):
        return ModelSource.TRANSFORMERS
    elif module_name.startswith("torchvision"):
        return ModelSource.TORCHVISION
    elif module_name.startswith("timm"):
        return ModelSource.TIMM
    else:
        return ModelSource.LOCAL


def source_name_to_enum(source_name):
    if source_name == "default":
        return ModelSource.DEFAULT
    elif source_name == "transformers":
        return ModelSource.TRANSFORMERS
    elif source_name == "torchvision":
        return ModelSource.TORCHVISION
    elif source_name == "timm":
        return ModelSource.TIMM
    elif source_name == "local":
        return ModelSource.LOCAL


def get_model(
    name: str,
    source: ModelSource = ModelSource.DEFAULT,
    pretrained=False,
    target_dataset="cifar10",
    **kwargs,
):
    # dataset_name, num_classes = re.findall(r"[a-z]+|\d+", target_dataset.lower())
    # if source == ModelSource.DEFAULT:
    #     ModelDict = MODEL_DICT
    # elif source == ModelSource.TRANSFORMERS:
    #     ModelDict = TRANSFORMER_MODEL_DICT
    # elif source == ModelSource.LOCAL:
    #     ModelDict = LOCAL_MODEL_DICT
    # elif source == ModelSource.TIMM:
    #     ModelDict = TIMM_MODEL_DICT
    # else:
    #     ModelDict = TRANSFORMER_MODEL_DICT
    print(name)
    model = MODEL_DICT[name]

    return model
