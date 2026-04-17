import os       # os.path.exists

 

"""

The most atomic way to train and run inference for a GPT in pure, dependency-free Python.

Modified version: Attention pattern changed from KV cache to basic matrix multiplication.

This implements the full attention matrix view as described in the transformer paper.

 

@karpathy

Modified for matrix-based attention computation

PyTorch version: Value scalar autograd replaced with torch tensors.

The only change is replacing the Value class (scalar-level graph) with

torch.nn.Parameter tensors. Each op (matmul, softmax, rmsnorm) is now a

single node in PyTorch's autograd graph regardless of tensor size, so

backward() is O(ops) instead of O(individual scalars).

"""

 

import time

import math

import random

import json

import csv

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset

from tqdm import tqdm

random.seed(42) # Let there be order among chaos

torch.manual_seed(42)

 

# ── Device / distributed setup ────────────────────────────────────────────

_USE_DDP = "LOCAL_RANK" in os.environ and torch.cuda.is_available()

if _USE_DDP:

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])

    world_size = dist.get_world_size()

    rank       = dist.get_rank()

    device     = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(device)

    is_main    = (rank == 0)

else:

    world_size = 1

    rank       = 0

    is_main    = True

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 

# Per-rank seed so each GPU samples different training windows

random.seed(42 + rank)

 

# Load ONLY the train split — the model must generalise to unseen pairs via

# digit-level representations, not memorisation.

dataset_train = load_dataset("akash-deep321/Product-Arithmetic", split="train")

if is_main:

    print(f"train docs: {len(dataset_train)}")

 

# ── Digit-level tokeniser ────────────────────────────────────────────────────

# Digits 0-9 map to themselves (token IDs 0-9).

# This gives the model numeric structure: similar digits → similar embeddings

# after training.  The model can then generalise to unseen (left, right) pairs.

STAR      = 10   # '*'

EQ        = 11   # '='

BOS       = 12

EOS       = 13

vocab_size = 14

# Sequence for 23*47=1081:

#   tokens : [BOS, 2, 3, STAR, 4, 7, EQ, 1, 0, 8, 1]   (11 tokens, positions 0-10)

#   targets: [2, 3, STAR, 4, 7, EQ, 1, 0, 8, 1, EOS]   (shift-by-1)

# Loss is computed only on the 4 result digits (target positions 6-9).

if is_main:

    print(f"vocab size: {vocab_size}  (digit-level)")

 

# Initialize the parameters, to store the knowledge of the model

n_layer = 4

n_embd = 256    # wider: must learn full 2-digit multiplication table

block_size = 11 # [BOS l1 l2 STAR r1 r2 EQ res1 res2 res3 res4] = 11 tokens

n_head = 4

head_dim = n_embd // n_head

 

# ── Model definition (nn.Module so DDP can sync gradients) ───────────────

def rmsnorm(x):

    # x: (T, n_embd) — normalises each row independently

    ms = (x * x).mean(dim=-1, keepdim=True)

    return x * (ms + 1e-5).rsqrt()

 

class ArithGPT(nn.Module):

    """Lightweight GPT for arithmetic, wrapped as nn.Module for DDP."""

    def __init__(self):

        super().__init__()

        std = 0.08

        self.wte     = nn.Parameter(torch.randn(vocab_size, n_embd) * std)

        self.wpe     = nn.Parameter(torch.randn(block_size, n_embd) * std)

        self.lm_head = nn.Parameter(torch.randn(vocab_size, n_embd) * std)

        for i in range(n_layer):

            setattr(self, f'layer{i}_attn_wq', nn.Parameter(torch.randn(n_embd, n_embd) * std))

            setattr(self, f'layer{i}_attn_wk', nn.Parameter(torch.randn(n_embd, n_embd) * std))

            setattr(self, f'layer{i}_attn_wv', nn.Parameter(torch.randn(n_embd, n_embd) * std))

            setattr(self, f'layer{i}_attn_wo', nn.Parameter(torch.randn(n_embd, n_embd) * std))

            setattr(self, f'layer{i}_mlp_fc1', nn.Parameter(torch.randn(4 * n_embd, n_embd) * std))

            setattr(self, f'layer{i}_mlp_fc2', nn.Parameter(torch.randn(n_embd, 4 * n_embd) * std))

 

    def forward(self, token_batch):

        """token_batch: (B, T) LongTensor"""

        dev = self.wte.device

        B, T = token_batch.shape

        pos  = torch.arange(T, device=dev)               # (T,)

 

        X = self.wte[token_batch] + self.wpe[pos]        # (B, T, n_embd)

        X = rmsnorm(X)

 

        causal_mask = torch.triu(

            torch.full((T, T), float('-inf'), device=dev), diagonal=1

        )  # (T, T) — broadcast over batch and heads

 

        for li in range(n_layer):

            wq  = getattr(self, f'layer{li}_attn_wq')

            wk  = getattr(self, f'layer{li}_attn_wk')

            wv  = getattr(self, f'layer{li}_attn_wv')

            wo  = getattr(self, f'layer{li}_attn_wo')

            fc1 = getattr(self, f'layer{li}_mlp_fc1')

            fc2 = getattr(self, f'layer{li}_mlp_fc2')

 

            # 1) Multi-head Attention

            X_residual = X

            X = rmsnorm(X)

            Q = X @ wq.T  # (B, T, n_embd)

            K = X @ wk.T

            V = X @ wv.T

 

            # (B, n_head, T, head_dim)

            Q = Q.view(B, T, n_head, head_dim).permute(0, 2, 1, 3)

            K = K.view(B, T, n_head, head_dim).permute(0, 2, 1, 3)

            V = V.view(B, T, n_head, head_dim).permute(0, 2, 1, 3)

 

            attn_logits  = Q @ K.transpose(-2, -1) / (head_dim ** 0.5)  # (B, n_head, T, T)

            attn_weights = F.softmax(attn_logits + causal_mask, dim=-1)

            X_attn = (attn_weights @ V).permute(0, 2, 1, 3).contiguous().view(B, T, n_embd)

            X = X_attn @ wo.T

            X = X + X_residual

 

            # 2) MLP

            X_residual = X

            X = rmsnorm(X)

            X = X @ fc1.T

            X = F.relu(X)

            X = X @ fc2.T

            X = X + X_residual

 

        return X @ self.lm_head.T  # (B, T, vocab_size)

 

# Instantiate model, move to device, wrap with DDP for multi-GPU training

model = ArithGPT().to(device)

if _USE_DDP:

    model = DDP(model, device_ids=[local_rank])

if is_main:

    print(f"num params: {sum(p.numel() for p in model.parameters())}")

    print(f"device: {device}  |  world_size: {world_size}")

 

# AdamW with strong weight decay — this is the key trigger for grokking

# (generalisation to unseen pairs).  Without weight decay transformers

# memorise the training set but never generalise.

learning_rate = 3e-4

weight_decay  = 1.0   # strong L2 as in Power et al. 2022 (Grokking paper)

optimizer = torch.optim.AdamW(

    model.parameters(), lr=learning_rate,

    betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay

)

 

batch_size   = 128

num_steps    = 1_000_000  

warmup_steps = 5_000

 

Loss = 0

loss_history = []       # (step, loss)

batch_loss_history = [] # (batch, avg_loss)

 

# Pre-parse training data

def to_digits(left, right, result):

    """Encode one multiplication example as digit tokens.

    tokens  = [BOS, l1, l2, STAR, r1, r2, EQ, d1, d2, d3, d4]  length 11

    targets = [l1,  l2, STAR, r1, r2, EQ, d1, d2, d3, d4, EOS] length 11

    Result is zero-padded to 4 digits (max 99*99=9801).

    """

    l1, l2 = left  // 10, left  % 10

    r1, r2 = right // 10, right % 10

    d1 = result // 1000

    d2 = (result // 100) % 10

    d3 = (result // 10)  % 10

    d4 =  result         % 10

    tokens  = [BOS, l1, l2, STAR, r1, r2, EQ, d1, d2, d3, d4]

    targets = [     l1, l2, STAR, r1, r2, EQ, d1, d2, d3, d4, EOS]

    return tokens, targets

 

parsed_dataset = []

for doc in dataset_train:

    l, r = doc["Expression"].split("*")

    parsed_dataset.append((int(l), int(r), int(doc["Result"])))

 

# Commutativity augmentation: a×b = b×a — doubles effective training data for free

parsed_dataset += [(r, l, res) for l, r, res in parsed_dataset if l != r]

random.shuffle(parsed_dataset)

if is_main:

    print(f"Training pairs (with commutativity): {len(parsed_dataset)}")

 

total_start = time.perf_counter()

 

for step in tqdm(range(num_steps), disable=not is_main):

    # Sample a batch

    batch   = random.choices(parsed_dataset, k=batch_size)

    tok_np  = [to_digits(l, r, res)[0] for l, r, res in batch]

    tgt_np  = [to_digits(l, r, res)[1] for l, r, res in batch]

    token_batch  = torch.tensor(tok_np, dtype=torch.long, device=device)   # (B, 11)

    target_batch = torch.tensor(tgt_np, dtype=torch.long, device=device)   # (B, 11)

 

    logits = model(token_batch)  # (B, 11, vocab_size)

 

    # Full-sequence loss — every position contributes gradient to the embeddings.

    # Predicting operand digits and structure tokens (STAR, EQ, BOS) is cheap

    # but forces the embedding layer to encode digit identity precisely, which

    # is what allows the model to generalise to unseen (left, right) pairs.

    loss = F.cross_entropy(

        logits.reshape(-1, vocab_size),

        target_batch.reshape(-1)

    )

 

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping

 

    # Linear warmup then constant LR

    lr_t = learning_rate * min(1.0, (step + 1) / warmup_steps)

    for pg in optimizer.param_groups:

        pg['lr'] = lr_t

    optimizer.step()

 

    # All-reduce loss across GPUs so rank-0 logs the true average

    if _USE_DDP:

        loss_reduced = loss.detach().clone()

        dist.all_reduce(loss_reduced, op=dist.ReduceOp.AVG)

        step_loss = loss_reduced.item()

    else:

        step_loss = loss.item()

 

    if is_main:

        loss_history.append((step, step_loss))

        Loss += step_loss

 

        LOG_EVERY = 500

        if (step + 1) % LOG_EVERY == 0:

            avg = Loss / LOG_EVERY

            batch_loss_history.append(((step + 1) // LOG_EVERY, avg))

            tqdm.write(f"Step {step+1:6d}/{num_steps} | loss {avg:.4f} | lr {lr_t:.2e}")

            Loss = 0

        elif step == num_steps - 1 and Loss > 0:

            remainder = (step + 1) % LOG_EVERY or LOG_EVERY

            avg = Loss / remainder

            batch_loss_history.append(((step + 1) // LOG_EVERY + 1, avg))

            tqdm.write(f"Step {step+1:6d}/{num_steps} | loss {avg:.4f} | lr {lr_t:.2e}")

            Loss = 0

 

elapsed = time.perf_counter() - total_start

if is_main:

    print(f"\nTotal training time : {elapsed:.1f}s")

    print(f"Per step            : {elapsed / num_steps * 1000:.1f}ms")

 

    # Unwrap DDP to access raw parameters for saving

    raw_model = model.module if _USE_DDP else model

    state_dict_data = {}

    for name, param in raw_model.named_parameters():

        # Map attribute names back to original dot-separated JSON key format

        # e.g. layer0_attn_wq → layer0.attn_wq

        if name.startswith('layer'):

            idx = name.index('_')

            key = name[:idx] + '.' + name[idx+1:]

        else:

            key = name

        state_dict_data[key] = param.detach().cpu().tolist()

    with open("Matrix-microGPT-2digit-torch.json","w") as f:

        json.dump(state_dict_data, f)

 

    # Save loss history to CSV

    with open("train_loss.csv", "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["step", "loss"])

        writer.writerows(loss_history)

    print("Loss history saved to train_loss.csv")

 

    with open("train_loss_batch.csv", "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["batch", "avg_loss"])

        writer.writerows(batch_loss_history)

    print("Batch loss history saved to train_loss_batch.csv")

 

    # Auto-plot loss curve

    import subprocess, sys

    subprocess.run([sys.executable, "plot_loss.py"], check=False)

 

if _USE_DDP:

    dist.destroy_process_group()

#——————————————————————————————————————————————————————————————————————#