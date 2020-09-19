# BayseLM
Bayesian Neural Network Language Model 

This is the script to build a toy Bayesian LSTM language model using PennTree Bank Dataset.


# SUDO:
requirements.txt

# Baseline LSTM
python main.py --train --data=data/ptb --model lstm --save ./ptblstmBaseline --no-tied --cuda 0 

# Bayesian LSTM start for scratch
python main.py --train --data=data/ptb --model lstm --gate-type bayes --position 0 --save ./ptblstmBayesP0scratch --no-tied --cuda 0

# Baysian LSTM using pretrained Prior and pretrained Model
python main.py --train --data=data/ptb --model lstm --gate-type bayes --position 0 --save ./ptblstmBayesP0_Pre --no-tied --cuda 0 --pretrain_prior

# Baysian LSTM using pretrained Prior, pretrained Model and reset Baysian Layer
python main.py --train --data=data/ptb --model lstm --gate-type bayes --position 0 --save ./ptblstmBayesP0_Pre_reset --no-tied --cuda 0 --pretrain_prior --reset_bayes
