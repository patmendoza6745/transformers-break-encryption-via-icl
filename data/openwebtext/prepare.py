"""
prepare.py
----------
1.  Downloads OpenWebText (≈8 M docs).
2.  Keeps only lowercase a‑z characters.
3.  Encodes each character with a tiny 29‑token vocabulary:
        26 letters  +  3 separators  ['|', ':', '?']
4.  Writes train.bin / val.bin (uint8) and meta.pkl (stoi / itos).
"""

import os, pickle, numpy as np
from tqdm import tqdm
from datasets import load_dataset  # huggingface
from multiprocessing import cpu_count

# ---------------------------------------------------------------------
# vocab ----------------------------------------------------------------
ALPHABET      = [chr(i) for i in range(ord('a'), ord('z') + 1)]
SEP_TOKENS    = ['|', '?']
VOCAB         = ALPHABET + SEP_TOKENS
stoi          = {ch: i for i, ch in enumerate(VOCAB)}
itos          = {i: ch for ch, i in stoi.items()}

encode_vec    = np.vectorize(lambda c: stoi[c], otypes=[np.uint8])

# ---------------------------------------------------------------------
num_proc_load = max(1, cpu_count() // 2)
num_proc_map  = num_proc_load
out_dir       = os.path.dirname(__file__) or '.'

def clean_and_encode(example):
    text = ''.join(ch for ch in example['text'].lower() if 'a' <= ch <= 'z')
    ids  = encode_vec(list(text))
    return {'ids': ids, 'len': len(ids)}

if __name__ == '__main__':
    ds   = load_dataset("openwebtext", num_proc=num_proc_load)
    split = ds['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split['val'] = split.pop('test')

    tokenized = split.map(
        clean_and_encode,
        remove_columns=['text'],
        desc='tokenizing',
        num_proc=num_proc_map
    )

    dtype = np.uint8
    for split_name, dset in tokenized.items():
        total_len = np.sum(dset['len'], dtype=np.uint64)
        filename  = os.path.join(out_dir, f'{split_name}.bin')
        arr       = np.memmap(filename, dtype=dtype, mode='w+', shape=(total_len,))
        shard_n   = 1024

        idx = 0
        for shard_id in tqdm(range(shard_n), desc=f'writing {filename}'):
            shard = dset.shard(num_shards=shard_n, index=shard_id, contiguous=True).with_format('numpy')
            chunk = np.concatenate(shard['ids'])
            arr[idx: idx + len(chunk)] = chunk
            idx += len(chunk)
        arr.flush()

    # -------- save vocab metadata
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump({'vocab_size': len(VOCAB), 'stoi': stoi, 'itos': itos}, f)

    print("Done. Files:", "train.bin", "val.bin", "meta.pkl")
