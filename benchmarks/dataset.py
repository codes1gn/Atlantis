import os
import re
from enum import Enum

import timm
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from transformers import ViTConfig, ViTForImageClassification, ViTModel
from transformers.models.resnet import (
    ResNetConfig,
    ResNetForImageClassification,
    ResNetModel,
)

from benchmarks.models.local import resnet as local_resnet

NORMALIZE_DICT = {
    "cifar10": dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    "cifar100": dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    "cifar10_224": dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    "cifar100_224": dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    "imagenet1k": dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "imagenet22k": dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
}


# use random data
class FakeDataset(Dataset):
    def __init__(self, len):
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        rand_image = torch.randn([3, 224, 224], dtype=torch.float32)
        label = torch.tensor(data=[index % 100], dtype=torch.int64)
        return rand_image, label


def get_fake_dataset():
    train_dst = FakeDataset(1281167)
    val_dst = FakeDataset(50000)
    test_dst = FakeDataset(50000)
    num_classes = 100
    input_size = (1, 3, 224, 224)
    return train_dst, val_dst, test_dst, num_classes, input_size


def get_dataset(name: str, data_root: str = "/data/workspace", return_transform=False):
    name = name.lower()
    data_root = os.path.expanduser(data_root)

    if name == "cifar10":
        num_classes = 10
        train_transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(**NORMALIZE_DICT[name]),
            ]
        )
        test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(**NORMALIZE_DICT[name]),
            ]
        )
        val_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(**NORMALIZE_DICT[name]),
            ]
        )
        data_root = os.path.join(data_root, "dataset")
        print(data_root)
        train_dst = datasets.CIFAR10(
            data_root, train=True, download=True, transform=train_transform
        )
        val_dst = datasets.CIFAR10(
            data_root, train=True, download=False, transform=val_transform
        )
        test_dst = datasets.CIFAR10(
            data_root, train=False, download=True, transform=test_transform
        )
        input_size = (1, 3, 32, 32)
    elif name == "cifar100":
        num_classes = 100
        train_transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(**NORMALIZE_DICT[name]),
            ]
        )
        test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(**NORMALIZE_DICT[name]),
            ]
        )
        data_root = os.path.join(data_root, "dataset")
        train_dst = datasets.CIFAR100(
            data_root, train=True, download=True, transform=train_transform
        )
        val_dst = datasets.CIFAR100(
            data_root, train=True, download=True, transform=test_transform
        )
        test_dst = datasets.CIFAR100(
            data_root, train=False, download=True, transform=test_transform
        )
        input_size = (1, 3, 32, 32)

    elif name == "imagenet1k":
        from torchvision.datasets import FakeData
        num_classes = 1000
        data_transform = T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.08, 1.0)),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(**NORMALIZE_DICT[name]),
            ]
        )
        test_transform = T.Compose(
            [
                T.RandomResizedCrop(224*(256 / 224)),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(**NORMALIZE_DICT[name]),
            ]
        )
        data_root = os.path.join(data_root, "dataset/imagenet")
        train_dst = FakeData(
            size=1281167,  # number of samples, adjust as needed
            image_size=(3, 224, 224),  # dimensions of ImageNet images
            num_classes=1000,  # number of classes in ImageNet1k
            transform=data_transform,
        )
        val_dst = FakeData(
            size=50000,  # number of samples, adjust as needed
            image_size=(3, 224, 224),  # dimensions of ImageNet images
            num_classes=1000,  # number of classes in ImageNet1k
            transform=test_transform,
        )
        test_dst = FakeData(
            size=50000,  # number of samples, adjust as needed
            image_size=(3, 224, 224),  # dimensions of ImageNet images
            num_classes=1000,  # number of classes in ImageNet1k
            transform=test_transform,
        )

        # legacy
        # train_dst = datasets.ImageNet(
        #     root=data_root, split="train", transform=data_transform
        # )
        # val_dst = datasets.ImageNet(
        #     root=data_root, split="train", transform=test_transform
        # )
        # test_dst = datasets.ImageNet(
        #     root=data_root, split="val", transform=test_transform
        # )

        # legacy 2
        # train_dst = datasets.ImageFolder(
        #     os.path.join(data_root, "train"), transform=data_transform
        # )
        # val_dst = datasets.ImageFolder(
        #     os.path.join(data_root, "val"), transform=data_transform
        # )
        # test_dst = datasets.ImageFolder(
        #     os.path.join(data_root, "val"), transform=data_transform
        # )
        input_size = (1, 3, 224, 224)
    elif name == "text":
        num_classes = 2
        input_size = (1, 3, 224, 224)
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_dst = dataset["train"]
        val_dst = dataset["validation"]
        test_dst = dataset["test"]
    elif name == "fake_dst":
        return get_fake_dataset()
    else:
        raise NotImplementedError
    if return_transform:
        return (
            train_dst,
            val_dst,
            test_dst,
            num_classes,
            input_size,
            train_transform,
            val_transform,
        )
    return train_dst, val_dst, test_dst, num_classes, input_size


def get_dataloader(
    name: str, batch_size: int, data_root: str = "/data/workspace", return_transform=False
):
    train_dst, val_dst, test_dst, num_classes = get_dataset(
        name, data_root, return_transform
    )[:4]

    train_loader = torch.utils.data.DataLoader(
        train_dst,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dst, batch_size=batch_size, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        val_dst, batch_size=batch_size, num_workers=4)
    return (
        train_loader,
        val_loader,
        test_loader,
        train_dst,
        val_dst,
        test_dst,
        num_classes,
    )


# import nltk
# import random
# import logging

# nltk.download("punkt")

# TOKENIZER_BATCH_SIZE = 20  # Batch-size to train the tokenizer on
# TOKENIZER_VOCABULARY = 25000  # Total number of unique subwords the tokenizer can have

# BLOCK_SIZE = 128  # Maximum number of tokens in an input sample
# NSP_PROB = 0.50  # Probability that the next sentence is the actual next sentence in NSP
# SHORT_SEQ_PROB = 0.1  # Probability of generating shorter sequences to minimize the mismatch between pretraining and fine-tuning.
# MAX_LENGTH = 256  # Maximum number of tokens in an input sample after padding

# MLM_PROB = 0.2  # Probability with which tokens are masked in MLM

# TRAIN_BATCH_SIZE = 2  # Batch-size for pretraining the model on
# MAX_EPOCHS = 1  # Maximum number of epochs to train the model for
# LEARNING_RATE = 1e-4  # Learning rate for training the model

# MODEL_CHECKPOINT = "bert-base-uncased"  # Name of pretrained model from 🤗 Model Hub

# from datasets import load_dataset

# dataset = load_dataset("wikitext")
# print(len(dataset["train"]))

# all_texts = [
#     doc for doc in dataset["train"]["text"] if len(doc) > 0 and not doc.startswith(" =")
# ]

# def batch_iterator():
#     for i in range(0, len(all_texts), TOKENIZER_BATCH_SIZE):
#         yield all_texts[i : i + TOKENIZER_BATCH_SIZE]

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# tokenizer = tokenizer.train_new_from_iterator(
#     batch_iterator(), vocab_size=TOKENIZER_VOCABULARY
# )

# dataset["train"] = dataset["train"].select([i for i in range(1000)])
# dataset["validation"] = dataset["validation"].select([i for i in range(1000)])

# # We define the maximum number of tokens after tokenization that each training sample
# # will have
# max_num_tokens = BLOCK_SIZE - tokenizer.num_special_tokens_to_add(pair=True)


# def prepare_train_features(examples):

#     """Function to prepare features for NSP task

#     Arguments:
#       examples: A dictionary with 1 key ("text")
#         text: List of raw documents (str)
#     Returns:
#       examples:  A dictionary with 4 keys
#         input_ids: List of tokenized, concatnated, and batched
#           sentences from the individual raw documents (int)
#         token_type_ids: List of integers (0 or 1) corresponding
#           to: 0 for senetence no. 1 and padding, 1 for sentence
#           no. 2
#         attention_mask: List of integers (0 or 1) corresponding
#           to: 1 for non-padded tokens, 0 for padded
#         next_sentence_label: List of integers (0 or 1) corresponding
#           to: 1 if the second sentence actually follows the first,
#           0 if the senetence is sampled from somewhere else in the corpus
#     """

#     # Remove un-wanted samples from the training set
#     examples["document"] = [
#         d.strip() for d in examples["text"] if len(d) > 0 and not d.startswith(" =")
#     ]
#     # Split the documents from the dataset into it's individual sentences
#     examples["sentences"] = [
#         nltk.tokenize.sent_tokenize(document) for document in examples["document"]
#     ]
#     # Convert the tokens into ids using the trained tokenizer
#     examples["tokenized_sentences"] = [
#         [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) for sent in doc]
#         for doc in examples["sentences"]
#     ]

#     # Define the outputs
#     examples["input_ids"] = []
#     examples["token_type_ids"] = []
#     examples["attention_mask"] = []
#     examples["next_sentence_label"] = []

#     for doc_index, document in enumerate(examples["tokenized_sentences"]):

#         current_chunk = []  # a buffer stored current working segments
#         current_length = 0
#         i = 0

#         # We *usually* want to fill up the entire sequence since we are padding
#         # to `block_size` anyways, so short sequences are generally wasted
#         # computation. However, we *sometimes*
#         # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
#         # sequences to minimize the mismatch between pretraining and fine-tuning.
#         # The `target_seq_length` is just a rough target however, whereas
#         # `block_size` is a hard limit.
#         target_seq_length = max_num_tokens

#         if random.random() < SHORT_SEQ_PROB:
#             target_seq_length = random.randint(2, max_num_tokens)

#         while i < len(document):
#             segment = document[i]
#             current_chunk.append(segment)
#             current_length += len(segment)
#             if i == len(document) - 1 or current_length >= target_seq_length:
#                 if current_chunk:
#                     # `a_end` is how many segments from `current_chunk` go into the `A`
#                     # (first) sentence.
#                     a_end = 1
#                     if len(current_chunk) >= 2:
#                         a_end = random.randint(1, len(current_chunk) - 1)

#                     tokens_a = []
#                     for j in range(a_end):
#                         tokens_a.extend(current_chunk[j])

#                     tokens_b = []

#                     if len(current_chunk) == 1 or random.random() < NSP_PROB:
#                         is_random_next = True
#                         target_b_length = target_seq_length - len(tokens_a)

#                         # This should rarely go for more than one iteration for large
#                         # corpora. However, just to be careful, we try to make sure that
#                         # the random document is not the same as the document
#                         # we're processing.
#                         for _ in range(10):
#                             random_document_index = random.randint(
#                                 0, len(examples["tokenized_sentences"]) - 1
#                             )
#                             if random_document_index != doc_index:
#                                 break

#                         random_document = examples["tokenized_sentences"][
#                             random_document_index
#                         ]
#                         random_start = random.randint(0, len(random_document) - 1)
#                         for j in range(random_start, len(random_document)):
#                             tokens_b.extend(random_document[j])
#                             if len(tokens_b) >= target_b_length:
#                                 break
#                         # We didn't actually use these segments so we "put them back" so
#                         # they don't go to waste.
#                         num_unused_segments = len(current_chunk) - a_end
#                         i -= num_unused_segments
#                     else:
#                         is_random_next = False
#                         for j in range(a_end, len(current_chunk)):
#                             tokens_b.extend(current_chunk[j])

#                     input_ids = tokenizer.build_inputs_with_special_tokens(
#                         tokens_a, tokens_b
#                     )
#                     # add token type ids, 0 for sentence a, 1 for sentence b
#                     token_type_ids = tokenizer.create_token_type_ids_from_sequences(
#                         tokens_a, tokens_b
#                     )

#                     padded = tokenizer.pad(
#                         {"input_ids": input_ids, "token_type_ids": token_type_ids},
#                         padding="max_length",
#                         max_length=MAX_LENGTH,
#                     )

#                     examples["input_ids"].append(padded["input_ids"])
#                     examples["token_type_ids"].append(padded["token_type_ids"])
#                     examples["attention_mask"].append(padded["attention_mask"])
#                     examples["next_sentence_label"].append(1 if is_random_next else 0)
#                     current_chunk = []
#                     current_length = 0
#             i += 1

#     # We delete all the un-necessary columns from our dataset
#     del examples["document"]
#     del examples["sentences"]
#     del examples["text"]
#     del examples["tokenized_sentences"]

#     return examples

# def get_tokenized_dataset():
#     tokenized_dataset = dataset.map(
#         prepare_train_features, batched=True, remove_columns=["text"], num_proc=1,
#     )
#     return tokenized_dataset
