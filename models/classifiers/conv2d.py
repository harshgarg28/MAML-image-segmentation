import torch
import torch.nn as nn

from .classifiers import register
from ..modules import *


__all__ = ['ConvClassifier']


@register('conv2d')
class ConvClassifier(Module):
  def __init__(self, in_dim, temp=1., learn_temp=False):
    super(ConvClassifier, self).__init__()
    self.in_dim = in_dim
    self.temp = temp
    self.learn_temp = learn_temp

    self.conv = nn.Conv2d(in_dim, 1, kernel_size=1, padding=0)

    if self.learn_temp:
      self.temp = nn.Parameter(torch.tensor(temp))

  def reset_parameters(self):
    nn.init.zeros_(self.linear.weight)
    nn.init.zeros_(self.linear.bias)

  def forward(self, x, params=None):
    assert x.dim() == 3
    segmentation = self.Conv2d(x_shot, get_child_dict(params, 'linear'))
    segmentation = segmentation * self.temp
    return segmentation