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
Python train.py
```

## Results

- Typically, language modeling is evaluated with ppl score of test dataset.
- All results below are evaluated at training epoch 20 except.
- The record of training process is under /Transformer/include/data/log.log.

| layers | pad_mask | epoch | dropout | accu | ppl  |
| ------ | -------- | ----- | ------- | ---- | ---- |
| 6      | True     | 20    | 0.3     | 25%  | 103  |
| 6      | True     | 20    | 0.2     | 25%  | 107  |
| 6      | True     | 20    | 0.0     | 23%  | 115  |
| 6      | False    | 20    | 0.2     | 22%  | 141  |

## Notes

- Change XL net to standard transformer.
- Change single gpu training code to multi-gpu.
- Change dropout into bayes.



## XL to standard

- Replace RelMultiHeadAttn with MultiHeadAttn.
- Reverse positional encoding.
- Change num_layers = 6, heads = 8, dim_model = 512.

### Usage

- STEP 1. Access to the destination folder, and PTB is under /data.

```
 cd /Standard-XL
```

- STEP 2. Run the command below to make datasets and train the model, and we can obtain results in .log file under /data.

```
Python train.py
```

### Results on ptb

| dropout | original | 0.0  | 0.1  | 0.2  |
| ------- | -------- | ---- | ---- | ---- |
| ppl     | 82       | 131  | 116  | 96   |



## Multi-gpu

- Use nn.DataParallel.
- Test: num_layers = 6, heads = 8, dim_model = 512, record time for each epoch, ppl and number of epochs of convergence.

### Usage

- STEP 1. Access to the destination folder, and PTB is under /data.

```
 cd /Standard-XL
```

- STEP 2. Run the command below for computer with two gpus.

```
 Python train.py --cuda --devices 01
```

### Results

| batch | single/s | ppl/epoch | multi | ppl/epoch |
| ----- | -------- | --------- | ----- | --------- |
| 10    | 91       | 92/10     | 71    | 110/35    |
| 20    | 56       | 104/10    | 62    | 120/40    |
| 40    | 41       | 106/15    | 60    | -         |
| 80    | 35       | 108/15    | 59    | -         |
| 160   | 32       | 110/15    | 59    | -         |

