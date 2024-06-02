import argparse
import copy
import gc
import logging
import os
from functools import partial
from typing import Any, Dict

import numpy as np
import registry
import torch
import torch.nn.functional as F
import torch_pruning as tp
from lightning import Trainer
from lightning.pytorch.callbacks import (
    Callback,
    DeviceStatsMonitor,
    ModelCheckpoint,
    Timer,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.strategies import DeepSpeedStrategy
from models import ClassifierModule, TransformerModule
from torch import nn

logging.disable(logging.INFO)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


parser = argparse.ArgumentParser()

# Basic options
parser.add_argument("--reversify", action="store_true", default=False)
parser.add_argument("--model", type=str, default="resnet")
parser.add_argument("--device", type=str, default="gpu")
# 0: fast dev 1: 5 epochs 2: default epochs
parser.add_argument("--version", type=str, default="v1")
parser.add_argument("--source", type=str, default="default")
parser.add_argument("--max-epochs", type=int, default=1000)
parser.add_argument("--fast-dev", action="store_true", default=False)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument('--checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    choices=["cifar10", "cifar100", "imagenet1k",
             "imagenet22k", "text", "fake_dst"],
)
parser.add_argument("--num_batches", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--group",
    type=str,
    default="perf",
    choices=["perf", "reversify", "sota"],
)
parser.add_argument(
    "--project",
    type=str,
    default="reversyn",
)

args = parser.parse_args()


def main():
    print(f"reversify:{args.reversify}")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    timer = Timer(duration="00:12:00:00")

    if (args.dataset in ["text"]):
        m = TransformerModule(
            args)
    else:
        m = ClassifierModule(
            args
        )
    # for accelarator in ["gpu", "cpu"]:
    print(args.device)
    model_name = f"{m.model_name}_{args.device}"
    # logger = TensorBoardLogger(
    #     save_dir="logs/performance/tensorboard",
    #     name=model_name,
    #     # log_graph=True,
    #     version=args.version,
    # )
    # TODO
    if not args.fast_dev:
        logger = WandbLogger(
            save_dir="logs/performance/wandb",
            log_model="best" if not args.fast_dev else False,
            entity="albertshi",
            name=model_name,
            project="reversyn",  # args.project,
            version=f"{model_name}_{args.version}",
            group=args.group,
        )

    trainer = Trainer(
        logger=logger if not args.fast_dev else None,
        limit_train_batches=args.num_batches if args.num_batches else 1.0,
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev,
        callbacks=[
            DeviceStatsMonitor(),
            # timer,
            # ModelCheckpoint(
            #     save_weights_only=True, mode="max", monitor="val_acc"
            # ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
        ],
        accelerator=args.device,
        # profiler=profiler,
    )
    # before training

    trainer.fit(model=m)
    # trainer.test(model=m)
    max_train_mem = torch.cuda.max_memory_allocated() / (1024**2)
    print(max_train_mem)
    # after training
    inference_mem = trainer.predict(model=m)
    # print(inference_time)
    print(inference_mem)

    # after training
    if not args.fast_dev:
        logger.experiment.config["batch_size"] = args.batch_size
        logger.experiment.config[
            "Image Size"
        ] = f"{m.example_input_array[0].shape[-1]}x{m.example_input_array[0].shape[-2]}"
        logger.experiment.config["Params"] = sum(
            p.numel() for p in m.model.parameters() if p.requires_grad
        ) / (1024**2)
        print("flops", m.flops)
        logger.experiment.config["FLOPs"] = m.flops
        print(
            f"train time: {timer.time_elapsed('train')} val time: {timer.time_elapsed('validate')} test time: {timer.time_elapsed('test')} "
        )
        logger.finalize("success")


if __name__ == "__main__":
    main()
