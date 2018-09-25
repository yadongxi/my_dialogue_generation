import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(args.embedding_dim, args.hidden_size, args.nlayers, args.dropout, bidirectional=args.bidirection)
        self.dropout = nn.Dropout(args.dropout)
        self.hidden_size = args.hidden_size
        self.nlayers = args.nlayers
        self.hidden_num = args.nlayers
        if args.bidirection:
            self.hidden_num = args.nlayers * 2

    def forward(self, embed, inputs, leng, hidden):
        inputs = embed(inputs).transpose(0, 1)
        inputs = pack_padded_sequence(inputs, leng)
        outputs, hidden = self.rnn(inputs, hidden)
        outputs, _ = pad_packed_sequence(outputs)
        hidden_c = self.dropout(torch.sum(hidden[0], dim=0, keepdim=True))
        hidden_h = self.dropout(torch.sum(hidden[1], dim=0, keepdim=True))
        return (hidden_c, hidden_h), (outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]).contiguous()

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters())
        return (weight.new_zeros(self.hidden_num, batch_size, self.hidden_size).to(device),
                weight.new_zeros(self.hidden_num, batch_size, self.hidden_size).to(device))


class Decoder(nn.Module):
    def __init__(self, args, attention):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(args.hidden_size, args.hidden_size, num_layers=1, bidirectional=False)
        self.attention = attention
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.hidden_size, args.ch_vocab_size)
        self.fc.bias.data.zero_()
        self.vocab_size = args.ch_vocab_size
        self.args = args
    def forward(self, args, embed, hidden, device, encode_output=None, mode=None, inputs=None):
        def rool_out(parts, hidden, encode_output, embed):
            ### 1 * batch
            length = len(parts)
            pools= []
            inputs = embed(parts[-1])
            for i in range(self.args.ch_seq_len - length):
                outputs, _ = self.rnn(inputs, hidden)
                if encode_output is not None:
                    hidden_c = self.attention(encode_output, hidden[0]) + hidden[0]
                    hidden_h = self.attention(encode_output, hidden[1]) + hidden[1]
                    hidden = (hidden_c, hidden_h)
                outputs = outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))
                logits = F.linear(outputs, embed.weight,  self.fc.bias).view(-1, self.vocab_size)
                word_weights = F.softmax(logits, dim=1)
                word_ix = torch.multinomial(word_weights, 1).transpose(0, 1)
                pools.append(word_ix)
                inputs = embed(word_ix)
            parts.extend(pools)
            return torch.cat(parts)
        if mode == "pre_train":
            ### teach force learn
            inputs = embed(inputs).transpose(0, 1)
            inputs = self.dropout(inputs)
            sents = []
            sents_index = []
            for i in range(self.args.ch_seq_len - 1):
                outputs, hidden = self.rnn(inputs[i:i + 1, :, :], hidden)
                if encode_output is None:
                    hidden_c = self.attention(encode_output, hidden[0]) + hidden[0]
                    hidden_h = self.attention(encode_output, hidden[1]) + hidden[1]
                    hidden = (hidden_c, hidden_h)
                outputs = outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))
                logits = F.linear(outputs, embed.weight, self.fc.bias)
                word_weights = F.softmax(logits, dim=1)
                word_ix = torch.multinomial(word_weights, 1)
                sents_index.append(word_ix)
                sents.append(logits)
            sents = torch.stack(sents).transpose(0, 1).contiguous()
            sents = sents.view(-1, self.vocab_size)
            return [sents, torch.cat(sents_index, dim=1)]
        elif mode == "train":
            ### MC search
            init = torch.ones((1, args.batch_size),dtype=torch.long).to(device)
            sents = [[[init] for _ in range(args.samples)] for _ in range(args.ch_seq_len - 1)]
            inputs = embed(init)
            distribution = []
            word_ind = []
            for i in range(0, self.args.ch_seq_len-1):
                outputs, hidden = self.rnn(inputs, hidden)
                if encode_output is not None:
                    hidden_c = self.attention(encode_output, hidden[0]) + hidden[0]
                    hidden_h = self.attention(encode_output, hidden[1]) + hidden[1]
                    hidden = (hidden_c, hidden_h)
                outputs = outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))
                logits = F.linear(outputs, embed.weight, self.fc.bias).view(-1, self.vocab_size)
                word_weights = F.softmax(logits, dim=1)
                distribution.append(word_weights)
                word_ix = torch.multinomial(word_weights, args.samples).transpose(0, 1)
                word_ind.append(word_ix[0])
                inputs = embed(word_ix[0:1])
                for j in range(i, args.ch_seq_len - 1):
                    if j == i:
                        cell = sents[i]
                        word_ix = word_ix.unsqueeze(1)
                        for k, piece in enumerate(word_ix):
                            cell[k].append(piece)
                            cell[k] = rool_out(cell[k], hidden, encode_output, embed)
                    else:
                        for cell in sents[j]:
                            cell.append(word_ix[0])
            return distribution, word_ind, sents
        elif mode == "evaluate":
            init = torch.ones((1, args.batch_size),dtype=torch.long).to(device)
            inputs = embed(init).to(device)
            word_ind = []
            for i in range(0, self.args.ch_seq_len-1):
                outputs, hidden = self.rnn(inputs, hidden)
                if encode_output is not None:
                    hidden_c = self.attention(encode_output, hidden[0]) + hidden[0]
                    hidden_h = self.attention(encode_output, hidden[1]) + hidden[1]
                    hidden = (hidden_c, hidden_h)
                outputs = outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))
                logits = F.linear(outputs, embed.weight, self.fc.bias).view(-1, self.vocab_size)
                word_weights = F.softmax(logits, dim=1)
                word_ix = word_weights.argmax(dim=1)
                word_ind.append(word_ix)
                inputs = embed(word_ix.unsqueeze(dim=0))            
            return torch.stack(word_ind).transpose(0, 1)
        else:
            print('some thing wrong')


class Attention(nn.Module):

    def __init__(self, args):
        super(Attention, self).__init__()
        self.fc = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fc.weight.size(1))
        self.fc.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, context):
        ## seq_len first, inputs seq_len * batch * dim, context batch * dim
        mid = inputs.clone()
        mid = self.fc(mid)
        mid = mid.view(inputs.size(0), -1)
        context = context.view(-1)
        weight = torch.mv(mid, context)
        weight = F.softmax(weight, dim=0).unsqueeze(1).unsqueeze(1)
        mid = inputs * weight
        return torch.sum(mid, dim=0)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.encoder = Encoder(args)
        self.attention = Attention(args)
        self.decoder = Decoder(args, self.attention)
        self.embed = nn.Embedding(args.en_vocab_size, args.embedding_dim)
        self.args = args

    def forward(self, inputs, device, mode):
        context, leng, response = inputs
        hidden = self.encoder.init_hidden(context.size(0), device)
        hidden, encoder_output = self.encoder(self.embed, context, leng, hidden)
        if mode == "pre_train":
            return self.decoder(self.args, self.embed,  hidden, device, encoder_output, mode, response)
        if mode == "train":
            return self.decoder(self.args, self.embed,  hidden, device, encoder_output, mode)
        if mode == "evaluate":
            return self.decoder(self.args, self.embed,  hidden, device, encoder_output, mode)

