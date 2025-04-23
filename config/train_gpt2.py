# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = False
wandb_project = 'CS182-final-project'
wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# each encryption key's prompt has 1024 plaintext/ciphertext pairs
# 256 batch size * 2048 block size * 1 gradient_accumulation_steps = 524,288
batch_size = 256
block_size = 2048
gradient_accumulation_steps = 1

# this makes total number of tokens be ~524M, with ~262M input/output pairs
max_iters = 1000
lr_decay_iters = 1000

# eval stuff
eval_interval = 100
eval_iters = 50
log_interval = 1

# weight decay
weight_decay = 1e-1
