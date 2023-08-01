from torch import arange
from transformers import ElectraModel
import torch
import torch.nn as nn

class Electra(nn.Module):
    def __init__(self, dr_rate):
        super(Electra, self).__init__()
        self.model = ElectraModel.from_pretrained("monologg/koelectra-small-v3-discriminator")
        self.dropout = nn.Dropout(dr_rate)

    def forward(self, token_idx, attention_mask):

        hidden_staes =  self.model(input_ids=token_idx, attention_mask=attention_mask.float().to(token_idx.device),
                              return_dict=False)

        # output = hidden_staes[:, 0, :] - for cls idx classifying


        return self.dropout(hidden_staes[0][:, 0, :])

        # return self.dropout(hidden_staes[1])

class Linear(nn.Module):
    def __init__(self, hidden_size=768):
        super(Linear, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()

        return sent_scores

class RNNClassifier(nn.Module):
    def __init__(self, bidirectional=True, num_layers=1, input_size=256,
                 hidden_size=256, dropout=0.1, num_class = 5):
        super(RNNClassifier, self).__init__()

        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions

        self.num_class = num_class

        self.rnn = nn.LSTM(
          input_size=input_size,
          hidden_size=self.hidden_size,
          num_layers=num_layers,
          bidirectional=bidirectional
          )

        self.wo = nn.Linear(self.num_directions * self.hidden_size, num_class, bias=True)
        self.dropout = nn.Dropout(dropout)

        if(num_class ==1):
            self.active_function = nn.Sigmoid()
        else:

            self.active_function = nn.Softmax(dim = 0)

    def forward(self, x):
        output, _ = self.rnn(x)
        out_fow = output[range(len(output)),  :self.hidden_size]
        out_rev = output[:, self.hidden_size:]
        output = torch.cat((out_fow, out_rev), 1)
        output = self.dropout(output)

        out_cls = self.active_function(self.wo(torch.squeeze(output, 1)))

        return out_cls

class Classifier(nn.Module):
    def __init__(self, argument_train):
        super(Classifier, self).__init__()

        self.electra = Electra(argument_train.drop_rate_bert)

        self.classifier = RNNClassifier(bidirectional=True, num_layers=argument_train.num_layer,
                                      input_size=argument_train.input_size, hidden_size=argument_train.hidden_size,
                                      dropout=argument_train.drop_rate_encoder, num_class=argument_train.num_class)

    def forward(self, token_idx, attn_mask):
        output = self.electra(token_idx, attn_mask)
        return self.classifier(output)


