# Transformer_testing

Inplementation for [Language Modeling with Deep Transformers](https://arxiv.org/pdf/1905.04226.pdf) (INTERSPEECH 2019) as I understood and relevant experiments on PTB. 

Given the inputs of word sequences, we can predict the next word in output layer. 

Preprocessing of input sequences has two steps, $Mask$ and $Embedding$. The main part is a stack of two layers, and each layer has as a two components: $self-attention$ and $feed-forward$ modules. 

The flowchart of this project is as follows.

<img src="/Users/collcertaye/WorkSpace/speech_recognition/Reports/transformer_modeling.png" alt="transformer_modeling" style="zoom:50%;" />

## Requirements

- python==3.x
- pytorch==1.3.0
- numpy>=1.15.4

## Usage

- STEP 1. Access to the destination folder, and PTB is under /data.

```
 cd /TransformerLM/include
```

- STEP 2. Run the command below to make datasets and train the model, and we can obtain results in .log file under /data.

```
Python train.py -- cuda 0
```

If you want to run the project with cpu, do this.

```
Python train.py -- cuda -1
```

## Results

- Typically, language modeling is evaluated with ppl score of test dataset.
- All results below are evaluated at training epoch 20.
- The record of training process is under /Transformer/include/data/log.log.

| layers | pos_emdedding | lr   | dropout | accu | ppl  |
| ------ | ------------- | ---- | ------- | ---- | ---- |
| 2      | True          | 1e-9 | 0.0     | 62%  | 12   |
| 6      | False         | 1e-9 | 0.0     | 22%  | 384  |
| 2      | False         | 1e-9 | 0.0     | 41%  | 46   |
| 2      | False         | 1e-6 | 0.0     | 37%  | 58   |
| 2      | False         | 1e-9 | 0.2     | 34%  | 68   |

## Notes

- I'm going to build subword-level language model.
- The $n-gram~language~model$ will be established soon.

