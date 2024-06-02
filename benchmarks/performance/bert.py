from reversyn import Compiler
from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import torch
from transformers import (
    AutoModelForSequenceClassification,
    BertConfig,
    BertForSequenceClassification,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.checkpoint import checkpoint
import torch.profiler as profiler
import time
import datasets
from datasets import load_dataset, load_from_disk

# datasets.config.HF_DATASETS_OFFLINE = True
dataset = load_from_disk("/data/workspace/dataset/yelp_review_full")
# dataset = load_dataset("yelp_review_full", cache_dir="/data/workspace/dataset/yelp_review_full")
# dataset.save_to_disk("/data/workspace/dataset/yelp_review_full")
print(dataset["train"][100])


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
small_train_dataset = tokenized_datasets["train"].shuffle(
    seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(
    seed=42).select(range(1000))


train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)


# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
model = BertForSequenceClassification(BertConfig())

# from torch.optim import AdamW

# optimizer = AdamW(model.parameters(), lr=5e-5)

# from transformers import get_scheduler

# num_epochs = 3
# num_training_steps = num_epochs * len(train_dataloader)
# lr_scheduler = get_scheduler(
#     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
# )


scaler = GradScaler()


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# tokens = next(iter(eval_dataloader))
# tokens = {k: v.to(device) for k, v in tokens.items()}
# input_ids = tokens['input_ids'].to(device)
# attention_mask = tokens['attention_mask'].to(device)
# labels = tokens['labels'].to(device)

batch = 14

input_ids = torch.ones((batch, 512), dtype=torch.long).to("cuda")
attention_mask = torch.ones((batch, 512), dtype=torch.long).to("cuda")
labels = torch.ones((batch), dtype=torch.long).to("cuda")

print(input_ids.shape)
print(attention_mask.shape)
print(labels.shape)
# tokens =tokens.requires_grad_()
# labels = torch.rand(8).to("cuda")
# labels,input_ids,token_type_ids,attention_mask
# del tokens['labels']
# # del tokens['input_ids']
# del tokens['token_type_ids']
# del tokens['attention_mask']

# print(tokens['input_ids'].shape)


model = Compiler(model, example_inputs=input_ids).reversify()


# torch.save(model, 'bert_rev.pth')

ll = []
tt = []
for i in range(3):
    with profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
    ) as prof:
        for i in range(3):
            start_time = time.perf_counter()
            model.train(True)
            # with autocast():
            model.gradient_checkpointing_enable()
            out = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
            # out = checkpoint(model, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # out = checkpoint(model, tokens['input_ids']).logits
            # loss = torch.nn.functional.cross_entropy(out, labels.long())
            loss = out.loss
            loss.requires_grad_(True)
            t3 = time.perf_counter()
            loss.backward()
            t4 = time.perf_counter()
            print(t4 - t3)
            tt.append(time.perf_counter() - start_time)
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
    print("cpu:", prof.key_averages().total_average().self_cpu_time_total_str)
    print("cuda:", prof.key_averages().total_average().self_cuda_time_total_str)
    ll.append(prof.key_averages().total_average().self_cuda_memory_usage)
print(sum(tt) / len(tt))
print(sum(ll) / len(ll))
