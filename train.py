import numpy as np
import torch

def word_decode(inputs, vocab):
    inputs = np.array(inputs).astype(np.int32)
    sents = []
    for i in range(inputs.shape[0]):
        tmp = []
        for index in inputs[i]:
            if index != 0:
                tmp.append(vocab.itos(index))
        sents.append(" ".join(tmp))
    return sents


class Train(object):
    def __init__(self, args, model, optim, loss, device, mode):
        self.model = model
        self.optimizer = optim
        self.loss = loss
        self.device = device
        self.mode = mode
    def train_step(self, inputs):
        ### only for generator
        inputs = [torch.tensor(x).to(self.device) for x in inputs]
        self.optimizer.zero_grad()
        outputs = self.model(inputs, self.device, self.mode)
        loss = self.loss(outputs, inputs)
        loss.backward()
        ### gradient clip
        self.optimizer.step()
        return outputs, loss
