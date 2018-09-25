from argparse import ArgumentParser

import torch
from data import Corpus
from models import Discriminator, Generator
from train import Train, word_decode


parser = ArgumentParser(description='all the related parameters')
# common parameters
parser.add_argument('--en_vocab_size', type=int, default=3500)
parser.add_argument('--ch_vocab_size', type=int, default=3500)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nlayers', type=int, default=1)
parser.add_argument('--bidirection', type=bool, default=True)
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
# discrinator parameters, later can add the l2 regulization
parser.add_argument('--dis_lr', type=float, default=0.0005)

# generator parameters
parser.add_argument('--gen_lr', type=float, default=0.001)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--samples', type=int, default=5)
# train parameter
parser.add_argument('--pre_train_epoch', type=int, default=30)
parser.add_argument('--train_epoch', type=int, default=20)
parser.add_argument('--save_path', type=str, default='save',help="path to saved model(to continue training)")
parser.add_argument('--gen_cycle', type=int, default=5)


parser.add_argument('--start_word', type=str, default='<sos>', help='start word for generation')
parser.add_argument('--seq_len', type=int, default=12, help='sequence length')
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()

if args.manualSeed is None:
   args.manualSeed = 24
torch.manual_seed(args.manualSeed)
device = torch.device("cuda:0")

corpus = Corpus("Data/train.txt",  
               test_path="Data/test.txt")
args.vocab_size = len(corpus.dictionary)
print("vocab_size", args.en_vocab_size)

# module init
discriminator = Discriminator(args).to(device)
generator = Generator(args).to(device)


def disc_loss(outputs, inputs):
   targets = torch.zeros((outputs.size(0), 2)).to(device)
   targets[:args.batch_size, 0] = 1
   targets[args.batch_size:, 1] = 1
   return torch.nn.BCELoss()(outputs, targets)


def pre_gen_loss(outputs, inputs):
   outputs = outputs[0]
   targets = inputs[2][:, 1:].contiguous()
   targets = targets.view(-1)
   return torch.nn.CrossEntropyLoss(ignore_index=0)(outputs, targets)


def gen_loss(outputs, inputs):
   distribution, word_ind, sample_sents = outputs
   base_rewards = []
   targ_rewards = []
   for step in sample_sents:
       tmp = torch.zeros(args.batch_size).to(device)
       for i, sent in enumerate(step):
           context, leng, response = inputs
           sent = sent.transpose(0, 1)
           reward = discriminator((context, response, sent), device)[args.batch_size:, 0]
           if i == 0:
               targ_rewards.append(reward)
           tmp += reward
       base_rewards.append(tmp / args.samples)
   i = 0
   loss = 0
   for word_distr, index in zip(distribution, word_ind):
       prob = torch.gather(word_distr, dim=1, index=index.unsqueeze(1))
       reward = targ_rewards[i] - base_rewards[i]
       loss += torch.sum(prob * reward)
   #max
   return -loss


disc_optim = torch.optim.Adam(discriminator.parameters(), weight_decay=0.0001, lr=args.dis_lr)
gen_optim = torch.optim.Adam(generator.parameters(), weight_decay=0.0001, lr=args.gen_lr)

disc_train = Train(args, discriminator, disc_optim, disc_loss, device, mode=None)
pre_generator_train = Train(args, generator, gen_optim, pre_gen_loss, device, mode="pre_train")   ## teach force
generator_train = Train(args, generator, gen_optim, gen_loss, device, mode="train")


print("pretrain begin")
for epoch in range(args.pre_train_epoch):
   batch_num = corpus.batch_num(args.batch_size)
   loss = 0
   d_loss = 0
   for batch_index in range(batch_num):
       context, leng, response = corpus.next_batch(args.batch_size)        
       fake_response, pre_gen_loss = pre_generator_train.train_step((context,  leng, response))
       loss += pre_gen_loss.item()
       # delete the sent start 
       response = response[:, 1:]
       fake_response_index = fake_response[1]
       _, dis_loss = disc_train.train_step((context, response, fake_response_index))
       d_loss += dis_loss.item()
   print("epoch:", epoch)
   print("pre_gen_loss:", loss / batch_num)
   print("dic_loss:", d_loss / batch_num)


print("adversarial train begin")
for epoch in range(args.train_epoch):
   batch_num = corpus.batch_num(args.batch_size)
   p_loss = 0
   d_loss = 0
   g_loss = 0
   for batch_index in range(batch_num):
       # train the generator
       context, leng, response = corpus.next_batch(args.batch_size)
       fake_response, gen_loss = generator_train.train_step((context, leng, response))
       g_loss += gen_loss.item()
       # teach force
       if batch_index % args.gen_cycle == 0:
           _, pre_gen_loss = pre_generator_train.train_step((context, leng, response))
           p_loss += pre_gen_loss.item()
       # train the discriminator
       fake_response_index = torch.stack(fake_response[1]).transpose(0, 1)
       response = response[:, 1:]
       _, dis_loss = disc_train.train_step((context, response, fake_response_index))
       d_loss += dis_loss.item()
   print("epoch:", epoch)
   print("pre_gen_loss:", p_loss / batch_num)
   print("gen_loss:", g_loss / batch_num)
   print("dic_loss:", d_loss / batch_num)

torch.save(discriminator, "save/discriminator.pt")
torch.save(discriminator, "save/generator.pt")
#
# test
generator.eval()
en_vocab = corpus.en_dictionary
ch_vocab = corpus.ch_dictionary
with open("test_result.txt", "w", encoding="utf-8") as f:
    for batch_index in range(corpus.batch_num(args.batch_size, mode="test")):
        # train the generator
        inputs = corpus.next_batch(args.batch_size, mode="test")
        inputs = [torch.tensor(x).to(device) for x in inputs]
        fake_response = generator(inputs, device, mode="evaluate")
        context = word_decode(inputs[0], en_vocab)[0]
        response = word_decode(inputs[2], ch_vocab)[0]
        fake = word_decode(fake_response, ch_vocab)[0]
        f.write(context + "\t")
        f.write(response + "\t")
        f.write(fake + "\n")