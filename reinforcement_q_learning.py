import gym
import math
import random
import numpy as np

import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvisions.transforms as T

env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

if is_ipython:
  from IPython import display

plt.ion()

# if GPU used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', 'state', 'action', 'next_state', 'reward')

class ReplayMemory(object):

  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.position = 0


