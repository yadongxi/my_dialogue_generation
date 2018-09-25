import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.hidden_size = args.hidden_size // 2
        self.seq_len = args.seq_len
        self.nlayers = args.nlayers
        self.drop = nn.Dropout(args.dropout)
        self.embed = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.sent_rnn = nn.LSTM(input_size=args.embedding_dim, hidden_size=self.hidden_size,
                          num_layers=self.nlayers, bidirectional=args.bidirection, dropout=args.dropout)
        self.para_rnn = nn.LSTM(input_size=args.hidden_size, hidden_size=args.hidden_size,
                                num_layers=1, bidirectional=args.bidirection, dropout=args.dropout)
        self.fc = nn.Linear(args.hidden_size*2, 2)
        if args.bidirection:
            self.direction = 2

    def forward(self, inputs, device, mode=None):
        ## inputs batch * seq
        context, response, fake_response = inputs

        batch_size = context.size(0)
        context = context.view(-1, self.seq_len)
        context = self.embed(context).transpose(0, 1)
        response = self.embed(response).transpose(0, 1)    
        fake_response = self.embed(fake_response).transpose(0, 1)
        hidden = self.init_hidden(batch_size, self.hidden_size, device, num=self.direction*self.nlayers)
        context_hidden = self.init_hidden(batch_size*(context.size(1)/batch_size), self.hidden_size, device, num=self.direction*self.nlayers)
        context_out, context_hidden = self.sent_rnn(context, context_hidden)
        response_out, response_hidden = self.sent_rnn(response, hidden)
        fake_out, fake_hidden = self.sent_rnn(fake_response, hidden)
        context_out = context_out[-1].view(-1, batch_size, self.hidden_size*2)
        real_para = torch.cat([context_out, response_out[-1:]], dim=0)
        fake_para = torch.cat([context_out, fake_out[-1:]], dim=0)
        para_hidden = self.init_hidden(batch_size, self.hidden_size * 2, device, num=self.direction)
        real_para_out, real_para_hidden = self.para_rnn(real_para, para_hidden)
        fake_para_out, fake_para_hidden = self.para_rnn(fake_para, para_hidden)
        outputs = torch.cat([real_para_out[-1], fake_para_out[-1]])
        outputs = self.drop(outputs)
        logits = self.fc(outputs)
        return F.softmax(logits, dim=1)

    def init_hidden(self, bsz, hidden_size, device, num=1):
        weight = next(self.parameters())
        return (weight.new_zeros(num, bsz, hidden_size).to(device),
                weight.new_zeros(num, bsz, hidden_size).to(device))
