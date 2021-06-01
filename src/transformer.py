""" RNN class wrapper, taking symbols, producing vector representations of prefixes

Sort of a vestigial part of more complex older code, but here in case we'd like
to hand-write RNNs again.
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import math

import utils
from transformers import GPT2Config, GPT2LMHeadModel

MAX_LEN = 6000
P_DROP = 0

class PytorchTransformerModel(nn.Module):
  """
  Class for mapping sequences of symbols to sequences
  of vectors representing prefixes, using PyTorch
  RNN classes.
  """

  def __init__(self, args, input_size, hidden_size, num_layers):
    super(PytorchTransformerModel, self).__init__()
    # self.input_size = input_size
    input_size = hidden_size
    self.input_size = hidden_size
    self.hidden_size = hidden_size
    self.vocab_size = args['language']['vocab_size']
    self.n_heads = args['lm']['num_heads']
    self.e_type = args['lm']['embedding_type']
    config = GPT2Config(n_embd=hidden_size, n_layer=num_layers, n_inner=hidden_size, attn_pdrop=P_DROP, embd_pdrop=P_DROP, resid_pdrop=P_DROP, vocab_size=self.vocab_size, n_head=self.n_heads, n_positions=MAX_LEN, n_ctx=MAX_LEN)
    print(config)
    self.model = GPT2LMHeadModel(config)
    print(self.model)
    self.device = args['device'] 
    self.model.to(self.device)
    tqdm.write('Constructing a GPT2 pytorch model w hidden size {}, layers {}, dropout {}'.format(hidden_size, num_layers, 0.0))
    
    if self.e_type == 'cos':
        funcs = [math.sin, math.cos]
        self.model.transformer.wpe.weight.data = torch.tensor([[ funcs[i % 2](pos / 10000 ** (2 * i / hidden_size)) 
            for i in range(hidden_size)] for pos in range(MAX_LEN)])
        self.model.transformer.wpe.weight.requires_grad=False
        
    if self.e_type == 'p' or self.e_type == 'pw':
        self.model.transformer.wpe.weight.data.zero_()
        self.model.transformer.wpe.weight.requires_grad=False

        self.embedding = nn.Embedding(self.vocab_size, input_size - 1)
        self.embedding.to(self.device)

        self.embedding_p = nn.Embedding(MAX_LEN, 1)
        self.embedding_p.weight.data = torch.tensor([[i/MAX_LEN] for i in range(MAX_LEN)])
        self.embedding_p.weight.requires_grad = False 
        self.embedding_p.to(self.device)

    if self.e_type == 'pw':
        k = args['language']['bracket_types']
        self.embedding = nn.Embedding(self.vocab_size, input_size - 1 - k - 4)
        self.embedding_e = nn.Embedding(self.vocab_size, k + 4)
        def get_row(i):
            arr = [0] * (k + 4)
            if i < 2 * k:
                arr[i % k] = arr[k + (i < k)] = 1
            else:
                arr[i - 2 * k + k + 2] = 1
            return arr
        self.embedding_e.weight.data = torch.tensor([get_row(i) for i in range(2 * k + 2)])
        self.embedding_e.weight.requires_grad = False 
        self.embedding.to(self.device)
        self.embedding_e.to(self.device)


  def forward(self, batch):
    """ Computes the forward pass to construct prefix representations.
    Arguments:
      batch: (batch_len, seq_len) vectors representing
             contexts
    Returns:
      hiddens: (batch_len, seq_len, hidden_size)
               recurrent state vectors for each token in input.
    """
    if self.e_type == 'default' or self.e_type == 'cos':
        return self.model.forward(batch).values()
    else:
        vec1 = self.embedding(batch)
        pos = torch.ones(batch.size(), device=self.device).cumsum(-1) - 1
        vec2 = self.embedding_p(pos.long())
        if self.e_type == 'p':
            vec = torch.cat((vec1, vec2), -1)
        else:
            vec3 = self.embedding_e(batch)
            vec = torch.cat((vec1, vec2, vec3), -1)
        return self.model.forward(inputs_embeds=vec).values()
