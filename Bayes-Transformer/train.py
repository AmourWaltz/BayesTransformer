import argparse
import os
import hashlib
import time
import math
import numpy as np
import torch
import torch.nn as nn

import data
from transformer_xl import AWDTransformerXL

from utils import batchify, get_batch, create_exp_dir, get_logger

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank Language Model')
parser.add_argument('--work_dir', default='TFM', type=str,
                    help='experiment directory.')
parser.add_argument('--data', type=str, default='../TransformerLM/include/data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='ptb',
                    help='location of dataset uesd')
parser.add_argument('--sentence-level', type=bool, default=True,
                    help='evaluating on sentence_level or segment_level for xl_net')

# model related
parser.add_argument('--n_layer', type=int, default=6,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=8,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=64,
                    help='head dimension')
parser.add_argument('--d_model', type=int, default=512,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=2048,
                    help='inner dimension in FF')
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='clamp length')

parser.add_argument('--dropoute', type=float, default=0.0,
                    help='dropout to remove words from embedding layer')
parser.add_argument('--dropouti', type=float, default=0.0,
                    help='dropout for input embedding vectors')
parser.add_argument('--dropouta', type=float, default=0.0,
                    help='dropout applied to multi-head attention layers')
parser.add_argument('--dropoutf', type=float, default=0.0,
                    help='dropout applied to positionwise ff layers')
parser.add_argument('--dropouth', type=float, default=0.0,
                    help='dropout applied to decoder layer output')
parser.add_argument('--dropouto', type=float, default=0.0,
                    help='dropout applied to the output (before the logit)')

# initialization
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='embedding initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.02,
                    help='parameters initialized by N(0, init_std)')

# optimization
parser.add_argument('--optimizer', type=str,  default='adam',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--lr', type=float, default=3e-4,
                    help='initial learning rate')
parser.add_argument('--lr_min', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--emb_mult', type=float, default=2,
                    help='multiplier for the learning rate of embeddings')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'const'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=3000,
                    help='warmup steps')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')

# regularization
parser.add_argument('--alpha', type=float, default=0.2,
                    help='alpha L2 regularization on activation')
parser.add_argument('--beta', type=float, default=0.1,
                    help='beta slowness regularization applied on activiation')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')

parser.add_argument('--std_epochs', type=int, default=10,
                    help='number of epochs with standard training')
parser.add_argument('--ema_epochs', type=int, default=0,
                    help='number of epochs with ema of params')
parser.add_argument('--decay_epochs', type=int, default=-1,
                    help='number of epochs with params decay')
parser.add_argument('--mu', type=float, default=-1,
                    help='mu used for EMA. set to -1 to use 1 / step.')
parser.add_argument('--epoch_ema', action='store_true',
                    help='do ema for each epoch. otherwise each step.')
parser.add_argument('--ema_lr_mult', type=float, default=0.5,
                    help='lr multiplier when switching to EMA.')

parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--ext_len', type=int, default=70,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous hidden')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--devices', type=int, default=1,
                    help='which GPU to use')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')

parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode')
parser.add_argument('--when', type=int, nargs='+', default=[],
                    help='when to save checkpoints')

# bayesian network
parser.add_argument('--bayes_ffn', type=int, default=0,
                    help='number of bayesian ffn layers')
parser.add_argument('--bayes_attn', type=int, default=0,
                    help='number of bayesian attn layers')
parser.add_argument('--bayes_embed', type=bool, default=True,
                    help='use bayes for embedding or not')


args = parser.parse_args()
args.tied = not args.not_tied
args.epochs = args.std_epochs + args.ema_epochs
if args.decay_epochs < 0:
    args.decay_epochs = args.std_epochs

if not args.resume:
    args.work_dir = os.path.join(args.work_dir, time.strftime("%Y%m%d-%H%M%S"))
    logging = create_exp_dir(args.work_dir, scripts_to_save=['train.py', 'transformer_xl.py'], debug=args.debug)
    if not args.debug:
        args.save = os.path.join(args.work_dir, args.save)
else:
    args.work_dir = os.path.join(*args.resume.split('/')[:-1])
    logging = get_logger(os.path.join(args.work_dir, 'log-resume.txt'),
                         log_=not args.debug)
    args.save = os.path.join(args.work_dir, 'model-resume.pt')

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device."
              "So you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda:
    torch.cuda.set_device(args.devices)

###############################################################################
# Load data
###############################################################################
def model_save(save_path):
    with open(save_path, 'wb') as f:
        torch.save([model, criterion, optimizer, scheduler], f)


def model_load(save_path):
    global model, criterion, optimizer, scheduler
    with open(save_path, 'rb') as f:
        model, criterion, optimizer, scheduler = torch.load(f)


eval_batch_size = 1
test_batch_size = 1

data_path = os.path.join(args.data, args.dataset)
if args.sentence_level is False:
    save_path = os.path.join(data_path, 'xl-net')
else:
    save_path = data_path
pass

# fn = os.path.join(
#     save_path,
#     'corpus.{}.data'.format(hashlib.md5(save_path.encode()).hexdigest()))
# if os.path.exists(fn):
#     logging('Loading cached dataset...')
#     corpus = torch.load(fn)
# else:
#     logging('Producing dataset...')
#     corpus = data.Corpus(args, data_path, eval_batch_size, test_batch_size)
#     torch.save(corpus, fn)

logging('Producing dataset...')
corpus = data.Corpus(args, data_path, eval_batch_size, test_batch_size)
# torch.save(corpus, fn)
# print(len(corpus.train_data), len(corpus.valid_data), len(corpus.test_data))

train_data = batchify(corpus.train_data, args.batch_size, args)
if args.sentence_level is False:
    val_data = batchify(corpus.valid_data, eval_batch_size, args)
    test_data = batchify(corpus.test_data, test_batch_size, args)
    pass
elif args.sentence_level is True:
    val_data = corpus.valid_loader
    test_data = corpus.valid_loader
    pass
else:
    val_data = None
    test_data = None
    pass
pass

args.max_decay_step = (train_data.size(0) + args.bptt - 1) // args.bptt * args.decay_epochs

###############################################################################
# Build the model
###############################################################################
# print(corpus.voc.idx2word)
ntokens = len(corpus.voc)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight'):
            if args.init == 'uniform':
                nn.init.uniform_(m.weight, -args.init_range, args.init_range)
            elif args.init == 'normal':
                nn.init.normal_(m.weight, 0.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('AdaptiveLogSoftmax') != -1:
        if args.init == 'uniform':
            nn.init.uniform_(m.cluster_weight, -args.init_range, args.init_range)
        elif args.init == 'normal':
            nn.init.normal_(m.cluster_weight, 0.0, args.init_std)
        nn.init.constant_(m.cluster_bias, 0.0)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_w_bias'):
            if args.init == 'uniform':
                nn.init.uniform_(m.r_w_bias, -args.init_range, args.init_range)
            elif args.init == 'normal':
                nn.init.normal_(m.r_w_bias, 0.0, args.init_std)
        if hasattr(m, 'r_r_bias'):
            if args.init == 'uniform':
                nn.init.uniform_(m.r_r_bias, -args.init_range, args.init_range)
            elif args.init == 'normal':
                nn.init.normal_(m.r_r_bias, 0.0, args.init_std)


model = AWDTransformerXL(ntokens, args.n_layer, args.n_head, args.d_model, args.d_head, args.d_inner,
                         args.dropoute, args.dropouti, args.dropouta, args.dropoutf, args.dropouth,
                         args.dropouto, tie_weight=args.tied, tgt_len=args.bptt, ext_len=args.ext_len,
                         mem_len=args.mem_len, clamp_len=args.clamp_len, bayes_ffn=args.bayes_ffn,
                         bayes_attn=args.bayes_attn, bayes_embed=args.bayes_embed).to(device)

model.apply(weights_init)
if args.emb_init == 'uniform':
    nn.init.uniform_(model.word_emb.weight, -args.init_range, args.init_range)
elif args.emb_init == 'normal':
    nn.init.normal_(model.word_emb.weight, 0.0, args.init_std)

criterion = nn.CrossEntropyLoss().to(device)

if args.resume:
    logging('Resuming model ...')
    model_load(args.resume)

params = list(model.parameters()) + list(criterion.parameters())
nonemb_params = [p for p in model.parameters() if p.size() != (ntokens, args.d_model)]
emb_params = list(model.word_emb.parameters())

args.total_params = sum(x.numel() for x in params if x is not None)
args.nonemb_params = sum(x.numel() for x in nonemb_params if x is not None)
args.emb_params = sum(x.numel() for x in emb_params if x is not None)

# optimizer
param_list = [nonemb_params, emb_params]
lr_list = [args.lr, args.lr * args.emb_mult]

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(
        [{'params': p, 'lr': lr} for p, lr in zip(param_list, lr_list)],
        weight_decay=args.wdecay)
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(
        [{'params': p, 'lr': lr} for p, lr in zip(param_list, lr_list)],
        weight_decay=args.wdecay)
if args.optimizer == 'asgd':
    optimizer = torch.optim.ASGD(
        [{'params': p, 'lr': lr} for p, lr in zip(param_list, lr_list)],
        t0=0, lambd=0., weight_decay=args.wdecay)

# scheduler
if args.optimizer != 'asgd':
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_decay_step-args.warmup_step,
                                                               eta_min=args.lr_min)
    elif args.scheduler == 'inv_sqrt':
        # originally used for Transformer (c.f. Attention is all you need)
        lr_lambda = lambda step: \
            step / args.warmup_step if step < args.warmup_step else \
            (args.warmup_step ** 0.5) / (step ** 0.5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lr_lambda)
    elif args.scheduler == 'const':
        lr_lambda = lambda step: \
            step / args.warmup_step if step < args.warmup_step else 1.
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lr_lambda)
else:
    scheduler = None

###
logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 100)


###############################################################################
# Training code
###############################################################################
# Statistics of correct words and total words in current batch.
def words_count(words):
    # count the words in current batchs that id is not PAD.
    non_pad_mask = words.ne(0)
    n_word = non_pad_mask.sum().item()
    return n_word


def evaluate(data_source, eval_bptt=50):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        eval_ext_len = args.ext_len + args.bptt - eval_bptt
        eval_mem_len = args.mem_len
        model.reset_length(eval_bptt, eval_ext_len, eval_mem_len)
    else:
        eval_mem_len = args.mem_len + args.bptt - eval_bptt
        eval_ext_len = args.ext_len
        model.reset_length(eval_bptt, eval_ext_len, eval_mem_len)

    mems = tuple()
    total_words = 0
    total_loss = 0

    if args.sentence_level is False:
        for i in range(0, data_source.size(0) - 1, eval_bptt):
            inputs, target = get_batch(data_source, i, args, seq_len=eval_bptt, ext_len=eval_ext_len)
            # print("data.size:" + str(data.size()) + "  target.size:" + str(target.size()))
            ret = model(inputs, target, *mems, return_h=True)
            raw_loss, mems, last_hid, _ = ret[0], ret[1:-2], ret[-2], ret[-1]
            raw_loss = raw_loss.mean()
            total_loss += target.size(0) * raw_loss.item()
            total_words = len(data_source)
        pass
    elif args.sentence_level is True:
        for _, (inputs, targets, sent_lens) in enumerate(data_source):
            # calculate loss and accuracy.
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
                pass

            # print("inputs.size:" + str(inputs.size()) + "  targets.size:" + str(targets.size()))
            ret = model(inputs, targets[1:, :], *mems, return_h=True,
                        sentence_level=args.sentence_level, sent_lens=sent_lens)
            raw_loss, mems, last_hid, _ = ret[0], ret[1:-2], ret[-2], ret[-1]
            raw_loss = raw_loss.mean()
            # total_loss += targets.size(0) * raw_loss.item()
            # total_words = len(data_source)
            n_words = words_count(inputs)
            total_loss += n_words * raw_loss.item()
            total_words += n_words
            pass
    pass

    # Switch back to the training mode
    model.reset_length(args.bptt, args.ext_len, args.mem_len)
    model.train()

    return total_loss / total_words


def train(data_display=False):
    # Turn on training mode which enables dropout.
    global step, ema, EMA
    total_loss = 0
    start_time = time.time()
    mems = tuple()
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence
        # length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        step += 1
        if scheduler is not None and not EMA and step <= args.max_decay_step:
            if args.scheduler == 'cosine':
                if step < args.warmup_step:
                    for loop_k in range(len(optimizer.param_groups)):
                        optimizer.param_groups[loop_k]['lr'] = \
                            lr_list[loop_k] * step / args.warmup_step
                else:
                    scheduler.step()
            else:
                scheduler.step()

        model.train()
        # print(train_data.size(), seq_len)
        inputs, targets = get_batch(train_data, i, args, seq_len=seq_len)

        optimizer.zero_grad()

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        # Forward
        # print(data, target)
        ret = model(inputs, targets, *mems, return_h=True, display=data_display)
        raw_loss, mems, last_hid, kl_loss = ret[0], ret[1:-2], ret[-2], ret[-1]
        raw_loss = raw_loss.mean()
        kl_loss = kl_loss / len(train_data)
        loss = raw_loss + kl_loss

        if data_display is True:
            data_display = False

        # Activiation Regularization
        if args.alpha:
            loss = loss + args.alpha * last_hid.pow(2).mean()

        # Temporal Activation Regularization (slowness)
        if args.beta:
            loss = loss + args.beta * (last_hid[1:] - last_hid[:-1]).pow(2).mean()

        # Backward
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem
        if args.clip:
            torch.nn.utils.clip_grad_norm_(params, args.clip)

        # optimizer step
        for loop_k in range(len(optimizer.param_groups)):
            optimizer.param_groups[loop_k]['lr'] *= (seq_len / args.bptt)
        optimizer.step()
        for loop_k in range(len(optimizer.param_groups)):
            optimizer.param_groups[loop_k]['lr'] /= (seq_len / args.bptt)

        # Keep exponential moving average of model parameters
        if EMA and not args.epoch_ema:
            if args.mu < 0:
                ema_mu = 1 / max(1, (step - ema_start_step))
            else:
                ema_mu = args.mu
            for p in model.parameters():
                ema[p].add_(p.data.sub(ema[p]).mul(ema_mu))

        # Logging
        total_loss += raw_loss.item()
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            log_str = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:.4g} ' \
                      '| ms/batch {:5.2f} | loss {:5.2f} | kl_loss {:5.2f} ' \
                      '| ppl {:8.2f} | bpt {:8.3f} '.format(epoch, batch, len(train_data) // args.bptt,
                                                            optimizer.param_groups[0]['lr'],
                                                            elapsed * 1000 / args.log_interval, cur_loss, kl_loss,
                                                            math.exp(cur_loss), cur_loss / math.log(2))
            logging(log_str)
            total_loss = 0
            start_time = time.time()
            # data_display = True
        #
        batch += 1
        i += seq_len


# Loop over epochs.
step = 0
lr = args.lr
stored_loss = float('inf')
ema, EMA = {}, False


# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        if epoch == args.std_epochs + 1:
            logging('Starting EMA at epoch {}'.format(epoch))

            EMA = True
            ema_start_step = step
            for p in model.parameters():
                ema[p] = p.data.clone()
            for k in range(len(optimizer.param_groups)):
                optimizer.param_groups[k]['lr'] *= args.ema_lr_mult
            args.save = os.path.join(args.work_dir, 'model-ema.pt')

        epoch_start_time = time.time()

        if epoch == args.std_epochs:
            display = False
            pass
        else:
            display = False
            pass
        pass

        train(data_display=display)

        # Evaluate using the EMA of parameters
        if EMA:
            if EMA and args.epoch_ema:
                if args.mu < 0:
                    ema_mu = 1 / epoch
                else:
                    ema_mu = args.mu
                for p in model.parameters():
                    ema[p].add_(p.data.sub(ema[p]).mul(ema_mu))

            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data.copy_(ema[prm])

            val_loss = evaluate(val_data, eval_batch_size)
            logging('-' * 89)
            logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} '
                    '| valid ppl {:9.2f} | valid bpt {:8.3f}'
                    .format(epoch, (time.time() - epoch_start_time),
                            val_loss, math.exp(val_loss), val_loss / math.log(2)))
            logging('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                logging('Saving Averaged!')
                stored_loss = val_loss

            for prm in model.parameters():
                prm.data.copy_(tmp[prm])

        # Evaluate using current model parameters
        else:
            val_loss = evaluate(val_data)
            logging('-' * 89)
            logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} '
                    '| valid ppl {:8.2f} | valid bpt {:8.3f}'
                    .format(epoch, (time.time() - epoch_start_time), val_loss,
                            math.exp(val_loss), val_loss / math.log(2)))
            logging('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                logging('Saving model (new best validation)')
                stored_loss = val_loss

        if epoch in args.when:
            model_save(os.path.join(args.work_dir, 'checkpoint.{}.pt'.format(epoch)))

except KeyboardInterrupt:
    logging('-' * 89)
    logging('Exiting from training early')

# Load the best saved model.
model_load(args.save)
model.temperature = 1.08

# Run on test data.
test_bptt = 1
test_loss = evaluate(test_data, test_bptt)

logging('=' * 89)
logging('| End of training | test loss {:5.2f} '
        '| test ppl {:8.2f} | test bpt {:8.3f}'.format(test_loss, math.exp(test_loss),
                                                       test_loss / math.log(2)))
logging('=' * 89)
