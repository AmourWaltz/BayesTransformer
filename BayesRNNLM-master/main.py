import argparse
import os, sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import gc
import data
import model as MODEL
from data import Corpus
from utils import device, repackage_hidden, create_exp_dir, save_checkpoint

parser = argparse.ArgumentParser(description='Pytorch Implementation of Bayesian Recurrent Neural Network Language Models')
parser.add_argument('--cuda',       type=int,   default=-1,     help='set cuda device id')

# Command Parameters
parser.add_argument('--train',      action='store_true',        help='train mode')
parser.add_argument('--pretrain_prior',    action='store_true',       help='using pretrain_priored mean as prior')
parser.add_argument('--reset_bayes',    action='store_true',       help='using pretrain_priored mean as prior')

# Dataset Parameters
parser.add_argument('--data',       type=str,   required=True,  help='choose the dataset: [ptb | swbd | callhm | ami]')
parser.add_argument('--save',       type=str,                   help='path to save the trained model')
parser.add_argument('--load',       type=str,                   help='path to load the stored model')

# Model Parameters
parser.add_argument('--model',      type=str,   default='gru',  help='model type: [lstm | gru]')
parser.add_argument('--embsize',    type=int,   default=200,    help='embeeding layer size, default: 300')
parser.add_argument('--hiddensize', type=int,   default=200,    help='hidden layer size, default: 300')
parser.add_argument('--nlayers',    type=int,   default=1,      help='number of hidden layers, default: 1')
parser.add_argument('--no-tied',    action='store_true',        help='tie the word embedding and softmax weights')
parser.add_argument('--gate-type',  type=str,   default='none', help='uncertain type: [none | gp | bayes | vrnn | vemb]')
parser.add_argument('--position',   type=int,   default=1,      help='uncertain position: [0-6]')



# Optimization Parameters
parser.add_argument('--lr',         type=float, default=4,      help='initial learning rate, default: 4')
parser.add_argument('--batchsize',  type=int,   default=10,     help='batch size, default: 32')
parser.add_argument('--alpha',      type=float, default=0,      help='alpha L2 regularization on RNN activation,default=1')
parser.add_argument('--beta',       type=float, default=0,      help='beta slowness regularization applied on RNN activiation, default=2')
parser.add_argument('--wdecay',     type=float, default=1.2e-6, help='weight decay applied to all weights')
parser.add_argument('--clip',       type=float, default=1.,     help='gradient clip, default: 0.25')
parser.add_argument('--seed',       type=int,   default=1,      help='random seed, default: 1')
parser.add_argument('--nepochs',    type=int,   default=500,    help='number of epochs for training, default: 500')
parser.add_argument('--log-period', type=int,   default=500,    help='period of log info, default: 500')

args = parser.parse_args()
flag=0

if args.cuda >= 0:
    torch.cuda.set_device(args.cuda)

create_exp_dir(args.save, scripts_to_save=['main.py', 'model.py', 'rnn.py'])

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if args.cuda < 0:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.set_device(args.cuda)
        torch.cuda.manual_seed_all(args.seed)

def logging(s, print_=True, log_=True):
    if print_:
        print(s)
        sys.stdout.flush()
    if log_:
        log_filename ='log.txt'
        with open(os.path.join(args.save, log_filename), 'a+') as f_log:
            f_log.write(s + '\n')

###############################################################################
# Load data
###############################################################################

dataset = args.data.split('/')[-1]
if dataset == 'ptb':
    args.ngram_ppl = 'data/ptb/4gram.ppl'
elif dataset == 'swbd':
    args.ngram_ppl = 'data/swbd/4gram.ppl'
    if args.rescore or args.interp:
        args.pma_txt = 'data/swbd/rescore/content.txt'
        args.pma_ppl = 'data/swbd/rescore/4gram.ppl'
        args.pma_dir = 'data/swbd/rescore'


corpus = Corpus(args.data, args.batchsize, args.batchsize, args.batchsize)
vocsize = corpus.voc.vocsize
ce_crit = nn.CrossEntropyLoss()


###############################################################################
# Build/Load the model
###############################################################################


if args.train:
    prior_dict=None
    prior=None
    if not args.pretrain_prior:
        model = MODEL.RNNLM(args.model, vocsize, args.embsize, args.hiddensize, args.nlayers,
                            args.no_tied, args.gate_type, args.position)
    else:
        final_type = args.gate_type
        args.gate_type = None
        model = MODEL.RNNLM(args.model, vocsize, args.embsize, args.hiddensize, args.nlayers,
                            args.no_tied, args.gate_type, args.position)
        if args.load!= None:
            prior = torch.load(os.path.join(args.load, 'model.pt'))
            model_dict = model.state_dict()
            prior_dict = prior.state_dict()
            prior_dict =  {k: v for k, v in prior_dict.items() if k in model_dict}
            model_dict.update(prior_dict)
            model.load_state_dict(model_dict)
            prior_dict = prior.state_dict()
            prior = prior.state_dict()
model = model.to(device)


logging('Cmd: python {}'.format(' '.join(sys.argv)))
total_params = sum(x.data.nelement() for x in model.parameters())
logging('Args: {}'.format(args))
logging('Vocabulary size: %s'%(vocsize))
logging('Model total parameters: {}'.format(total_params))
print(model)

###############################################################################
# Training code
###############################################################################

def train(optimizer, piror=None):
    model.train()

    total_loss, total_cnt = 0, 0
    start_time = time.time()
    hidden = model.init_hidden(args.batchsize)
    for i, (inputs, targets, sent_lens) in enumerate(corpus.train_loader):
        targets_packed = pack_padded_sequence(targets, sent_lens)[0]
        model.zero_grad()
        log_prob, raw_outputs = model(
                inputs, hidden, sent_lens)
        raw_loss = ce_crit(log_prob.view(-1, vocsize), targets_packed.view(-1))
        loss = raw_loss
        kl = 0.
        if(args.gate_type == 'bayes'):
            loss = loss + model.rnns[0].kl_divergence(prior)/corpus.train_data.n_sents*args.batchsize
            kl = kl + model.rnns[0].kl_divergence(prior)/corpus.train_data.n_sents*args.batchsize
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        n_words = torch.sum(sent_lens).type(torch.FloatTensor)
        total_loss += raw_loss.item() * n_words
        total_cnt += n_words
        kl_loss = kl * n_words
        if (i % args.log_period == 0 and i > 0):
            cur_loss = total_loss/total_cnt
            kl_loss = kl_loss /total_cnt
            elapsed = time.time() - start_time
            if (args.gate_type == 'vrnn' or args.gate_type == 'vemb'):
                cur_kld_loss = tota_kld_loss/total_cnt
                logging('| epoch {:3d} | {:4d}/{:4d} batches | lr {:02.1e} | ms/batch {:5.2f} | '
                        'loss {:4.2f} | kld_loss {:4.2f} | ppl {:8.2f}'.format(
                    epoch, i, corpus.train_data.n_sents//args.batchsize, optimizer.param_groups[0]['lr'],
                    elapsed*1000/args.log_period, cur_loss, cur_kld_loss, math.exp(cur_loss)))
            else:
                logging('| epoch {:3d} | {:4d}/{:4d} batches | lr {:02.1e} | ms/batch {:5.2f} | '
                        'loss {:4.2f}| kl loss: {:4.2f}| ppl {:8.2f}'.format(
                    epoch, i, corpus.train_data.n_sents//args.batchsize, optimizer.param_groups[0]['lr'],
                    elapsed*1000/args.log_period, cur_loss, kl_loss, math.exp(cur_loss)))
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    hidden = model.init_hidden(args.batchsize)
    total_loss, total_cnt = 0, 0
    with torch.no_grad():
        for chunk, (inputs, targets, sent_lens) in enumerate(dataloader):
            targets_packed = pack_padded_sequence(targets, sent_lens)[0]
            log_prob, _ = model(inputs, hidden, sent_lens)
            loss = ce_crit(log_prob.view(-1, vocsize), targets_packed.view(-1))
            n_words = torch.sum(sent_lens).type(torch.FloatTensor)
            total_loss += loss.item() * n_words
            total_cnt += n_words
    return total_loss/total_cnt


try:    
    if args.train:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
        stored_loss = np.Infinity

        best_val_loss = []
        best_state = None
        val_loss, count_div, tol_div = np.Infinity, 0, 1e-3
        init_lr = optimizer.param_groups[0]['lr']

        for epoch in range(1, args.nepochs+1):
            epoch_start_time = time.time()
            train(optimizer,prior)

            val_loss = evaluate(corpus.valid_loader)
            logging('-' * 89)
            logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            logging('-' * 89)

            if val_loss < stored_loss:
                best_state = model.state_dict()
                save_checkpoint(model, optimizer, args.save)
                logging('Saving Normal!')
                stored_loss = val_loss
                count_div = 0
            else:
                model.load_state_dict(best_state)
                count_div += 1

            if count_div > 0:
                if optimizer.param_groups[0]['lr']*2**5 < init_lr:
                    logging('Done!')
                    break
                elif args.pretrain_prior and flag==0:
                    flag = 1
                    print(" Finish pretrain_prioring and using Bayesian Net! ")
                    args.gate_type = final_type
                    prior_dict = model.state_dict() # 1
                    prior_update_dict =  {k: v.clone() for k, v in prior_dict.items()} #2
                    model = MODEL.RNNLM(args.model, vocsize, args.embsize, args.hiddensize, args.nlayers,
                                        args.no_tied, args.gate_type, args.position)
                    model = model.to(device)
                    model_dict = model.state_dict()

                    if args.reset_bayes:
                        print("Reset Bayes")
                        model_update_dict = {k: v for k, v in model_dict.items()}
                        model_dict.update(prior_dict)
                        if 1 <= args.position <= 4:
                            print(prior_dict['rnns.0.theta_hh_mean'][(args.position-1)*args.hiddensize:args.position*args.hiddensize])
                            print(prior_update_dict['rnns.0.theta_hh_mean'][(args.position-1)*args.hiddensize:args.position*args.hiddensize])
                            print(model_dict['rnns.0.theta_hh_mean'][(args.position-1)*args.hiddensize:args.position*args.hiddensize])
                            print(model_update_dict['rnns.0.theta_hh_mean'][(args.position-1)*args.hiddensize:args.position*args.hiddensize])
                            model_dict['rnns.0.theta_hh_mean'][(args.position-1)*args.hiddensize:args.position*args.hiddensize] = model_update_dict['rnns.0.theta_hh_mean'][(args.position-1)*args.hiddensize:args.position*args.hiddensize]
                            print(model_dict['rnns.0.theta_hh_mean'][(args.position-1)*args.hiddensize:args.position*args.hiddensize])
                            print(prior_update_dict['rnns.0.theta_hh_mean'][(args.position-1)*args.hiddensize:args.position*args.hiddensize])
                            print(prior_dict['rnns.0.theta_hh_mean'][(args.position-1)*args.hiddensize:args.position*args.hiddensize])
                            model_dict['rnns.0.theta_ih_mean'][(args.position-1)*args.hiddensize:args.position*args.hiddensize] = model_update_dict['rnns.0.theta_ih_mean'][(args.position-1)*args.hiddensize:args.position*args.hiddensize]
                        if args.position == 0:
                            model_dict['rnns.0.theta_hh_mean']=model_update_dict['rnns.0.theta_hh_mean']
                            model_dict['rnns.0.theta_ih_mean']=model_update_dict['rnns.0.theta_ih_mean']                                
                    else:
                        model_dict.update(prior_dict)
                    
                    model.load_state_dict(model_dict)
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr/4., weight_decay=args.wdecay)
                    prior = prior_update_dict
                    
                else:
                    count_div = 0
                    logging('Lower Learning Rate!')
                    optimizer.param_groups[0]['lr'] /= 4.
            best_val_loss.append(val_loss)
except KeyboardInterrupt:
    logging('-' * 89)
    logging('Exiting from training early')

# Load the best saved model.
model = torch.load(os.path.join(args.save, 'model.pt'))
if args.cuda >= 0:
    model = model.to(device)
else:
    model = model

# Run on test data.
test_loss = evaluate(corpus.test_loader)
logging('=' * 89)
logging('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logging('=' * 89)


