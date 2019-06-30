#-*-coding:utf-8-*-
import torch
from torchcrf import CRF
num_tags = 5  # number of tags is 5
model = CRF(num_tags)
seq_length = 3  # maximum sequence length in a batch
batch_size = 2  # number of samples in the batch
emissions = torch.randn(seq_length, batch_size, num_tags)
tags = torch.tensor([[0, 1], [2, 4], [3, 1]], dtype=torch.long)  # (seq_length, batch_size)
model(emissions, tags)
# mask size is (seq_length, batch_size)
# the last sample has length of 1
mask = torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.uint8)
print(model(emissions, tags, mask=mask))
