import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder,TransformerEncoderLayer

from empchat.datasets.tokens import PAD_TOKEN

class TransformerAdapter(nn.Module)
	def __init__ (self, opt, dictionary):
		super(TransformerAdapter, self).__init__()
		self.opt = opt
		self.pad_idx = dictionary[PAD_TOKEN]
		self.embeddings = nn.Embedding(len(dictionary), opt.embeddings_size, padding_idx=self.pad_idx)
		nn.init.normal_(self.embeddings.weight, mean=0, std=0.05)
		self.encoder_layer = TransformerEncoderLayer(d_model = opt.transformer_dim,nhead = opt.transformer_n_head,dim_feedforward = 4*opt.transformer_dim)
		self.ctx_encoder = TransformerEncoder(self.encoder_layer,opt.n_layers)