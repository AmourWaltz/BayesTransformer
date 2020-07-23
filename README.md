# TransformerLM

Inplementation for [Language Modeling with Deep Transformers](https://arxiv.org/pdf/1905.04226.pdf) (INTERSPEECH 2019) as I understood and relevant experiments on PTB. 

Given the inputs of word sequences, we can predict the next word in output layer. 

Preprocessing of input sequences has two steps, *Mask* and *Embedding*. The main part is a stack of two layers, and each layer has as a two components: *self-attention* and *feed-forward* modules. 

The flowchart of this project is as follows.

<img src="TransformerLM/include/data/modeling transformer.png" alt="modeling transformer" style="zoom:50%;" />

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
- All results below are evaluated at training epoch 20 except 3-layer transformer.
- The record of training process is under /Transformer/include/data/log.log.

| layers | pos_enc | epoch | dropout | accu | ppl  |
| ------ | ------- | ----- | ------- | ---- | ---- |
| 2      | True    | 20    | 0.2     | 22%  | 173  |
| 2      | False   | 20    | 0.2     | 23%  | 168  |
| 3      | False   | 35    | 0.2     | 20%  | 194  |
| 2      | False   | 20    | 0.0     | 19%  | 206  |

## Notes

- Change singe gpu training code to multi-gpu.
- Change dropout into bayes.

