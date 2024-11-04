# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html

import logging
import os
import time
import urllib.request
from types import SimpleNamespace
from urllib.error import HTTPError

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from deepspeed.profiling.flops_profiler import FlopsProfiler
from torch import Tensor

from benchmarks import registry
from reversyn import MetaRunner


class ClassifierModule(L.LightningModule):
    def __init__(
        self,
        configs,
    ):
        """CIFARModule.

        Args:
            name: Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams: Hyperparameters for the model, as dictionary.
            optimizer_name: Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Create model
        model = registry.get_model(
            configs.model, source=registry.source_name_to_enum(configs.source))
        # source = registry.source_name_to_enum(registry.get_model_source(model))
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.train_dst,
            self.val_dst,
            self.test_dst,
            num_classes,
        ) = registry.get_dataloader(name=configs.dataset, batch_size=configs.batch_size, data_root="/data/workspace")
        L.seed_everything(42)
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Example input for visualizing the graph in Tensorboard
        # print(next(iter(self.train_loader)).shape)
        images, labels = next(iter(self.train_loader))
        # print(next(iter(self.train_loader))[0].shape)
        # print(next(iter(self.train_loader))[1].shape)
        # print(next(iter(self.val_loader))[0].shape)
        self.example_input_array = images

        if not configs.reversify:
            self.model = model
        else:
            self.model = MetaRunner(
                model, self.example_input_array, device=torch.device("cpu")
            ).run()
            # self.model = model
        self.source = registry.get_model_source(self.model)
        model_name = f"{self.source.value}_{configs.model}_{configs.dataset}_bs{configs.batch_size}"
        self.model_name = f"reversify_{model_name}" if configs.reversify else f"baseline_{model_name}"
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        self.flops = None
        self.flops_profiler = FlopsProfiler(self.model)
        # Predict
        self.starter, self.ender = torch.cuda.Event(
            enable_timing=True
        ), torch.cuda.Event(enable_timing=True)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        output = self.model(imgs)
        if self.source == registry.ModelSource.TRANSFORMERS:
            output = output.logits
        return output

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if hasattr(self, "hparams") and hasattr(self.hparams, "optimizer_name"):
            if self.hparams.optimizer_name == "Adam":
                # AdamW is Adam with a correct implementation of weight decay (see here
                # for details: https://arxiv.org/pdf/1711.05101.pdf)
                optimizer = optim.AdamW(
                    self.parameters(), **self.hparams.optimizer_hparams
                )
            elif self.hparams.optimizer_name == "SGD":
                optimizer = optim.SGD(
                    self.parameters(), **self.hparams.optimizer_hparams
                )
            else:
                assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'
        else:
            optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [scheduler]

    def common_step(self, batch, batch_idx, stage=None):
        begin = time.time()
        imgs, labels = batch
        logits = self(imgs)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct / imgs.shape[0]

        if stage:
            self.log(f"{stage}_loss", loss)
            self.log(f"{stage}_acc", accuracy)
            self.log(f"{stage}_time", time.time() - begin)
            self.log(f"{stage}_mem", torch.cuda.memory_allocated())
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        loss, accuracy = self.common_step(batch, batch_idx, "train")
        return loss  # Return tensor to call ".backward" on

    # def training_epoch_end(self, outputs):
    #     self.accuracy.reset()

    def validation_step(self, batch, batch_idx):
        self.common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.common_step(batch, batch_idx, "test")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_loader

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mem_start = torch.cuda.memory_allocated() / float(1024**2)
        images, _ = batch
        self.starter.record()
        _ = self(images)
        self.ender.record()
        # wait for gpu sync
        torch.cuda.synchronize()
        inference_time = self.starter.elapsed_time(self.ender) * 1e-3
        # self.log("inference_time", inference_time)
        # 计算内存占用量
        mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # 转换为MB
        memory_cached = torch.cuda.memory_cached() / 1024 / 1024  # 转换为MB
        mem_used = mem_allocated - mem_start
        torch.cuda.empty_cache()
        return mem_used

    def on_train_start(self) -> None:
        self.flops_profiler.start_profile()
        return super().on_train_start()

    def on_train_end(self) -> None:
        self.flops = self.flops_profiler.get_total_flops(True)
        # self.flops_profiler.print_model_profile()
        self.flops_profiler.end_profile()
        return super().on_train_end()

    def on_before_backward(self, loss: Tensor) -> None:
        torch.cuda.reset_max_memory_allocated()
        return super().on_before_backward(loss)

    def on_after_backward(self) -> None:
        self.log(f"bwd_mem", torch.cuda.memory_allocated())
        self.log(f"bwd_mem_max", torch.cuda.max_memory_allocated())
        return super().on_after_backward()


if __name__ == "__main__":
    model = ClassifierModule("resnet", reversify=True)
    trainer = L.Trainer()
    trainer.fit(model)
