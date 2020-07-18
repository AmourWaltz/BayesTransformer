import math
import time
from data import Corpus
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as func
import torch.optim as optim
import torch.utils.data
from model import Modeling

if torch.cuda.is_available():
    device = torch.device('cuda')
    pass
else:
    device = torch.device('cpu')
    pass
pass

PAD = 0
epochs = 20
log_path = "data/"
log_flag = True

criterion = nn.CrossEntropyLoss()


# Statistics of correct words and total words in current batch.
def accu_evaluation(pred, gold):
    # transform gold and pred has the same shape on dim 0, 1.
    gold = gold.contiguous().view(-1)
    voc_size = pred.size()[-1]
    pred = pred.contiguous().view(-1, voc_size)

    # get max index of pred on dim=2, count the correct words_id with gold.
    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    # count the words in current batchs that id is not PAD.
    non_pad_mask = gold.ne(PAD)
    n_word = non_pad_mask.sum().item()

    return n_word, n_correct


# Calculate cross entropy loss, apply label smoothing if needed.
def loss_calculation(pred, gold, sent_lens, smoothing=False):
    vocab_size = pred.size()[-1]
    if smoothing:
        eps = 0.1
        # transpose targets dataset and cut first line
        pred = pred[:, 1:, :]
        # transform gold.size(): (lens, batch) to (batch, lens)
        gold = gold.t()[:, 1:]

        # transform pred.size(): (batch, lens, vocab) to (batch * lens, vocab)
        # transform gold.size(): (batch, lens) to (batch * lens)
        # print(gold.size(), pred.size())
        gold = gold.contiguous().view(-1)
        pred = pred.contiguous().view(-1, vocab_size)
        # print(gold.size(), pred.size())

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)

        # torch.cuda.available, cuda cannot convert Tensor to numpy()
        if torch.cuda.is_available():
            one_hot = one_hot.cpu().numpy() * (1 - eps) + (1 - one_hot.cpu().numpy()) * eps / (vocab_size - 1)
            pass
        else:
            one_hot = one_hot.numpy() * (1 - eps) + (1 - one_hot.numpy()) * eps / (vocab_size - 1)
            pass
        pass
        log_prb = func.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(PAD)
        if torch.cuda.is_available():
            loss = -(torch.tensor(one_hot).cuda() * log_prb).sum(dim=1)
            pass
        else:
            loss = -(torch.tensor(one_hot) * log_prb).sum(dim=1)
            pass
        pass
        loss = loss.masked_select(non_pad_mask).mean()
        pass
    else:
        # pack_padded_sequence() to ignore the paddings.
        pred_packed = pack_padded_sequence(pred.transpose(0, 1), sent_lens)[0]
        targets_packed = pack_padded_sequence(gold, sent_lens)[0]
        loss = criterion(pred_packed.view(-1, vocab_size), targets_packed.view(-1))
        pass
    pass

    return loss


# A simple wrapper class for learning rate scheduling.
class ScheduledOptim(nn.Module):
    def __init__(self, optimizer, d_model, n_warmup_steps=4000):
        super(ScheduledOptim, self).__init__()
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    # Step with the inner optimizer.
    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    # Zero out the gradients by the inner optimizer
    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    # Learning rate scheduling per step.
    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            pass
        pass


# Epoch operation in training phase.
def train_epoch(model, training_data, optimizer, smooth_flag):
    model.train()
    total_loss = 0
    total_words = 0
    total_correct = 0

    for loop_i, (inputs, targets, sent_lens) in enumerate(training_data):
        # backward
        optimizer.zero_grad()
        pred, _ = model(inputs.t(), sent_lens)
        word_number, correct_words = accu_evaluation(pred[:, 1:, :], targets.t()[:, 1:])
        loss = loss_calculation(pred, targets, sent_lens, smoothing=smooth_flag)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        total_words += word_number
        total_loss += loss.item() * word_number

        total_correct += correct_words
        accuracy = total_correct / total_words

        if loop_i % 1000 == 0:
            print("train_loop: {loop: 5d}, loss: {loss: 8.5f}, accuracy: {accu:3.3f}".
                  format(loop=loop_i, loss=loss, accu=accuracy * 100))
            # print("pred output: ", pred)
            # print("pred: ", pred.argmax(dim=2))
            # print("targets: ", targets.t())
            # print("pred, targets: ", pred_packed, pred_packed.argmax(dim=1), targets_packed)
            pass
        pass
    pass

    loss_per_word = total_loss / total_words
    accuracy = total_correct / total_words
    return loss_per_word, accuracy


# Epoch operation in evaluation phase.
def val_epoch(model, valid_data):
    model.eval()
    with torch.no_grad():
        val_correct, val_words, total_loss = val_test_iteration(model, valid_data)
    pass
    loss_per_word = total_loss / val_words
    accuracy = val_correct / val_words
    return loss_per_word, accuracy


# Training process.
def train_process(model, training_data, validation_data, optimizer, smooth_flag):
    model.to(device)
    log_file = log_path + 'log.log'

    print('[Info] Results will be written to file: {}'.format(log_file))

    with open(log_file, 'w') as log:
        log.write('Every second should be created. {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        pass
    pass

    for epoch_i in range(epochs):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer, smooth_flag)
        train_time = (time.time() - start) / 60
        print('  - (Train) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, elapse: {elapse:3.3f} min'.format(
            ppl=math.exp(train_loss), accu=100 * train_accu, elapse=train_time))

        start = time.time()
        valid_loss, valid_accu = val_epoch(model, validation_data)
        val_time = (time.time() - start) / 60
        print('  - (Valid) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, elapse: {elapse:3.3f} min'.format(
            ppl=math.exp(valid_loss), accu=100 * valid_accu, elapse=val_time))

        if log_flag:
            with open(log_file, 'a') as log:
                log.write('[ Epoch: {epoch: 5d}] \n'.format(epoch=epoch_i))
                log.write('  - (Train) loss: {loss: 8.5f}, ppl: {ppl: 8.5f}, accu: {accu:3.3f}, '
                          'elapse: {elapse:3.3f} min \n'.format(loss=train_loss, ppl=math.exp(train_loss),
                                                                accu=100 * train_accu, elapse=train_time))
                log.write('  - (Valid) loss: {loss: 8.5f}, ppl: {ppl: 8.5f}, accu: {accu:3.3f}, '
                          'elapse: {elapse:3.3f} min \n'.format(loss=valid_loss, ppl=math.exp(valid_loss),
                                                                accu=100 * valid_accu, elapse=val_time))
                pass
            pass
        pass
    pass

    return model


# Validation and testing dataset iterations.
def val_test_iteration(model, data):
    total_words = 0
    total_loss = 0
    total_correct = 0
    for _, (inputs, targets, sent_lens) in enumerate(data):
        # calculate loss and accuracy.
        pred, _ = model(inputs.t(), sent_lens)
        word_number, correct_words = accu_evaluation(pred[:, 1:, :], targets.t()[:, 1:])
        loss = loss_calculation(pred, targets, sent_lens)

        total_words += word_number
        total_loss += loss.item() * word_number

        total_correct += correct_words
    pass

    return total_correct, total_words, total_loss


# Tesing process.
def test_process(model, test_data):
    total_loss = 0
    test_words = 0
    test_correct = 0
    log_file = log_path + 'log.log'
    print('[Info] Testing performance will be written to file: {}'.format(log_file))

    for epoch_i in range(epochs):
        with torch.no_grad():
            test_correct, test_words, total_loss = val_test_iteration(model, test_data)
        pass
    pass

    test_loss = total_loss / test_words
    test_accu = test_correct / test_words
    print(' - (Test) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %'.format(
        ppl=math.exp(test_loss), accu=100 * test_accu))

    if log_flag:
        with open(log_file, 'a') as log:
            log.write('[ Test: ]   loss: {loss: 8.5f}, ppl: {ppl: 8.5f}, accu: {accu:3.3f} \n'
                      .format(loss=test_loss, ppl=math.exp(test_loss), accu=100 * test_accu))
            pass
        pass
    pass


def main():
    print("Every second should be created.", time.strftime('%Y-%m-%d %H:%M:%S'))

    # Preparing DataLoader.
    corpus = Corpus('data/ptb', train_batch_size=8, valid_batch_size=16, test_batch_size=1)

    # Loading Dataset and set parameters.
    training_data, validation_data, testing_data = corpus.train_loader, corpus.valid_loader, corpus.test_loader
    vocab_size = len(corpus.voc)
    max_length = corpus.max_length
    num_layers, model_dim, num_heads, ffn_dim, dropout, lr, smooth_flag = 2, 512, 8, 2048, 0.0, 1e-9, False

    print("[Data] train_length:", len(corpus.train_data), ", val_length:", len(corpus.valid_data),
          ", test_length:", len(corpus.test_data), ", vocab_size:", vocab_size, ", max_length:", max_length)

    print("[Para]  num_layers:", num_layers, ", model_dim:", model_dim, ", num_heads:", num_heads,
          ", ffn_dim:", ffn_dim, ", dropout:", dropout, ", learning_rate:", lr, ", smoothing_flag:", smooth_flag)

    transformer = Modeling(src_vocab_size=vocab_size, src_max_len=max_length,
                           num_layers=num_layers, model_dim=model_dim, num_heads=num_heads,
                           ffn_dim=ffn_dim, dropout=dropout).to(device)

    optimizer = ScheduledOptim(optim.Adam(
        filter(lambda x: x.requires_grad, transformer.parameters()),
        betas=(0.9, 0.98), eps=lr),
        model_dim)

    model = train_process(transformer, training_data, validation_data, optimizer, smooth_flag)
    test_process(model, testing_data)
    pass


if __name__ == '__main__':
    main()
