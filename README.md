# CS182 Final Project

Instructions on how to setup below. Also, give `train.py` a read. It will make understanding what's going on a lot easier.

## Install dependencies
```
conda env create -f environment.yml
conda activate 182env
```

## Reproducing GPT-2

Tokenize [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```sh
python3 data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the token ids in one sequence, stored as raw uint8 bytes. Then we're ready to kick off training. To reproduce (take a look at `train.sh` for more detailed and relevant arguments), simply run

```sh
bash run_train.sh config/mono_alphabetic_sub.yaml 1 single-gpu 4 logs/test.log
bash run_train.sh config/vigenere.yaml 1 single-gpu 4 logs/test.log
```
within `~/CS182-project`. Note that the default config values pertain to the best model results found in the paper.