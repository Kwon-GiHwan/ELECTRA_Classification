import torch
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup as cosine_warmup
from torch import nn
from tqdm import tqdm

import numpy as np

import math

class Trainer(nn.Module):
    def __init__(self, model, device, dset_train, dset_test, arg_train = None):
        super(Trainer, self).__init__()

        self.batch_size = arg_train.batch_size
        self.warmup_rate = arg_train.warmup_rate
        self.epoch = arg_train.epoch
        self.grad_norm = arg_train.grad_norm
        self.learn_rate = arg_train.learn_rate

        self.dset_train = dset_train
        self.dset_test = dset_test

        self.device = device
        self.model = model

        self.no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_param = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(self.optimizer_param, lr=self.learn_rate)
        if(arg_train.num_class == 2):
            self.loss_fn = nn.BCELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.train_len = len(self.dset_train) * self.epoch
        self.warmup_step = int(self.train_len * self.warmup_rate)

        self.scheduler = cosine_warmup(self.optimizer, num_warmup_steps=self.warmup_step, num_training_steps=self.train_len)

    def calc_acc(self, X, Y):
        max_vals, max_indices = torch.max(X, 1)
        train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
        return train_acc

    def train_loop(self):

        self.model.train()
        train_acc = 0.0

        for batch_id, (_input_idx, _attention_mask, _label) in enumerate(tqdm(self.dset_train)):
            self.optimizer.zero_grad()

            input_idx = _input_idx.to(self.device)
            attention_mask = _attention_mask.to(self.device)
            label = _label.to(self.device)

            output = self.model(input_idx, attention_mask)

            loss = self.loss_fn(output.squeeze(), label.float())
            loss.requires_grad_(True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)  # gradient clipping
            self.optimizer.step()
            self.scheduler.step()

            train_acc += self.calc_acc(output, label.float())

        print(
            "train loop: loss {}  acc {}".format(loss.data.cpu().numpy(), train_acc / len(self.dset_train)))

    def test_loop(self):

        self.model.eval()
        with torch.no_grad():
            test_loss = 0.0
            test_acc = 0.0

            for batch_id, (_input_idx, _attention_mask, _label) in enumerate(tqdm(self.dset_test)):
                input_idx = _input_idx.to(self.device)
                attention_mask = _attention_mask.to(self.device)
                label = _label.to(self.device)

                output = self.model(input_idx, attention_mask)

                test_loss += self.loss_fn(output, label.float())

                test_acc += self.calc_acc(output, label.float())

            print(
                "test loop: loss {}  acc {}".format( test_loss.data.cpu().numpy() / len(self.dset_test), test_acc / len(self.dset_test)))

    def valid_loop(self):

        self.model.eval()
        with torch.no_grad():
            valid_result = []

            for batch_id, (_input_idx, _attention_mask) in enumerate(tqdm(self.dset_test)):
                input_idx = _input_idx.to(self.device)
                attention_mask = _attention_mask.to(self.device)

                output = self.model(input_idx, attention_mask)

                values, indices = torch.max(output)
                valid_result.append(indices)
                torch.cuda.empty_cache()

            return valid_result
