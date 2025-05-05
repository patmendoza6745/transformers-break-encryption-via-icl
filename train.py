import yaml
import os
import argparse

import torch.distributed as dist
import torch
import wandb
import numpy as np
import pickle
import time
import re

from schemes import *
from torch.nn.parallel import DistributedDataParallel
from contextlib import nullcontext
from model import GPTConfig, GPT

# Vocab
ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]
SEP_BAR, SEP_Q = '|', '?'

# Model architecture
N_LAYER = 12
N_HEAD = 8
N_EMBD = 256
DROPOUT = 0.0
BIAS = False

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def merge_configs(base_config, override_args):
    for key, value in vars(override_args).items():
        if value is not None:
            base_config[key] = value
    return base_config

def get_batch(split, config, data_dir, device_type, device, stoi, alpha_ids):
    mmap = np.memmap(os.path.join(data_dir, f'{split}.bin'),
                     dtype=np.uint8, mode='r')
    
    block_size = config['block_size']
    batch_size = config['batch_size']
    scheme_type = config['scheme_type']

    k_pairs = block_size // 2                     # desired number of pairs
    known_k   = k_pairs - 1                       # last one is the query
    prompt_sz = 2 * k_pairs                       # block_size * 2 tokens (e.g., 1024 * 2 = 2048 tokens)
    assert prompt_sz == block_size, "block_size must be 2*k_pairs"

    X = torch.full((batch_size, block_size - 1), stoi[SEP_BAR],  dtype=torch.long)
    Y = torch.full((batch_size, block_size - 1), -1,          dtype=torch.long)

    for b in range(batch_size):
        
        if scheme_type == 'mono-alphabetic-sub':
            # ----- 1. grab k plaintext letters from corpus -------------------
            start = np.random.randint(0, len(mmap) - k_pairs - 1)
            plain = mmap[start:start + k_pairs].copy()          # np.uint8, shape (k_pairs,)

            # ----- 2. fresh random key for this sample -----------------------
            scheme = MonoAlphabetic(alpha_ids=alpha_ids)
        elif scheme_type == 'vigenere':
             # ----- 1. grab k plaintext letters from corpus -------------------
            start = np.random.randint(0, len(mmap) - k_pairs - 1)
            plain = mmap[start:start + k_pairs].copy()          # np.uint8, shape (k_pairs,)

            # ----- 2. fresh random key for this sample -----------------------
            # ----- randomly sample between 4 and 32 -----------------------
            key_length = np.random.randint(4, 32 + 1)
            scheme = Vigenere(key_length=key_length, alpha_ids=alpha_ids)
        else:
            raise ValueError("Invalid scheme type")
        
        # ----- 3. build prompt ------------------------------------------
        buf, tgt = [], []
        for i, p in enumerate(plain):
            c = scheme.enc(p)
            if i < known_k:                                 # give answer
                buf.extend([c, p])
                tgt.extend([p, -1])
            else:                                           # query pair
                buf.extend([c])
                tgt.extend([p])

        X[b] = torch.from_numpy(np.asarray(buf,  np.uint8))
        Y[b] = torch.from_numpy(np.asarray(tgt, np.int64))

    if device_type == 'cuda':
        X, Y = X.pin_memory().to(device, non_blocking=True), \
               Y.pin_memory().to(device, non_blocking=True)
    else:
        X, Y = X.to(device), Y.to(device)

    return X, Y

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, warmup_iters, lr_decay_iters, min_lr, learning_rate):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(config, data_dir, device_type, device, stoi, alpha_ids, ctx, model):
    out = {}
    eval_iters = config['eval_iters']
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, config, data_dir, device_type, device, stoi, alpha_ids)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(config):
    wandb_log = config['wandb_log']
    wandb_project = config['wandb_project']
    wandb_run_name = config['wandb_run_name']

    base_out_dir = config['base_out_dir']

    batch_size = config['batch_size']
    block_size = config['block_size']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    weight_decay = config['weight_decay']
    learning_rate = config['learning_rate']
    grad_clip = config['grad_clip']

    warmup_iters = config['warmup_iters']
    max_iters = config['max_iters']
    lr_decay_iters = config['lr_decay_iters']
    min_lr = config['min_lr']
    decay_lr = config['decay_lr']

    eval_interval = config['eval_interval']
    log_interval = config['log_interval']
    save_interval = config['save_interval']
    eval_only = config['eval_only']

    dataset = config['dataset']

    backend = config['backend']
    compile = config['compile']
    scheme_type = config['scheme_type']

    device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

    config_str = ', '.join(f"{k}={v}" for k, v in config.items())
    print(f"\n==== Final Config ====\n{config_str}\n=======================\n")

    if wandb_log:
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
        print(f"W&B logging enabled: Project={wandb_project}, Run={wandb_run_name}")
    else:
        print("W&B logging disabled")

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        dist.init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    out_dir = f"{base_out_dir}/{wandb_run_name}"

    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        files = os.listdir(out_dir)
        files = [f for f in files if f.endswith('.pt')]
        if files:
            init_from = "resume"
        else:
            if master_process:
                os.makedirs(out_dir, exist_ok=True)

    # set seed
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # poor man's data loader
    data_dir = os.path.join('data', dataset)
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    vocab_size = meta['vocab_size'] 
    print(f"Using vocab size of {vocab_size} (a-z + separators)", flush=True)

    alpha_ids = np.array([stoi[c] for c in ALPHABET], dtype=np.uint8)

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    total_tokens = 0
    best_val_loss = 1e9

    # model init
    model_args = dict(n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, block_size=block_size,
                    bias=BIAS, vocab_size=vocab_size, dropout=DROPOUT)
    
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch...")

        # Determine the vocab size we'll use for from-scratch training
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}...")

        def get_latest_ckpt(directory):
            max_token_count = -1
            ckpt_file = None
            pattern = re.compile(r'.*?(\d+(\.\d+)?)M_ckpt\.pt$')

            for filename in os.listdir(directory):
                match = pattern.search(filename)
                if match:
                    token_count = int(float(match.group(1)))
                    if token_count > max_token_count:
                        max_token_count = token_count
                        ckpt_file = filename

            return ckpt_file 

        # resume training from a checkpoint.
        ckpt_file = get_latest_ckpt(out_dir)
        ckpt_path = os.path.join(out_dir, ckpt_file)
        print(f"Using checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']

        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]

        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']

        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        state_dict = None

        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        total_tokens = checkpoint['total_tokens']

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value

    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # adamW optimizer
    beta1 = 0.9
    beta2 = 0.95

    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)", flush=True)
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # logging
    if wandb_log and master_process:
        config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
        wandb_config = {k: globals()[k] for k in config_keys}
        wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)

    # training loop
    X, Y = get_batch('train', config, data_dir, device_type, device, stoi, alpha_ids) # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    while True:
        curr_mem = torch.cuda.memory_allocated() / 2e30
        peak_mem = torch.cuda.max_memory_allocated() / 2e30

        total_tokens += tokens_per_iter

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, min_lr, learning_rate) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(config, data_dir, device_type, device, stoi, alpha_ids, ctx, model)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)
            if wandb_log:
                wandb.log({
                    "tokens": total_tokens,
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                    "peak_memory": peak_mem,
                    "state_memory": curr_mem,
                })
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
        if iter_num == 0 and eval_only:
            break

        if iter_num % save_interval == 0 and iter_num > 0 and master_process:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
                'total_tokens': total_tokens
            }

            token_count = int(total_tokens // 1e6)
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'{wandb_run_name}_{token_count}M_ckpt.pt'))

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train', config, data_dir, device_type, device, stoi, alpha_ids)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%", flush=True)
            if wandb_log:
                wandb.log({
                    "tokens": total_tokens,
                    "iter": iter_num,
                    "train/loss": lossf,
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                    "peak_memory": peak_mem,
                    "state_memory": curr_mem,
                })
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
                'total_tokens': total_tokens
            }
            token_count = int(total_tokens // 1e6)
            print(f"saving final checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'final_{wandb_run_name}_{token_count}M_ckpt.pt'))
            break

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script')

    # YAML config file
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')

    # Optional overrides
    parser.add_argument('--learning_rate', type=float, help='Learning rate override')
    parser.add_argument('--batch_size', type=int, help='Batch size override')
    parser.add_argument('--base_out_dir', type=str, help='Output directory override')

    args = parser.parse_args()

    config = load_config(args.config)

    final_config = merge_configs(config, args)

    train(final_config)
