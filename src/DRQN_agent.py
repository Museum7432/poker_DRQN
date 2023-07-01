import os
import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy

from .models.DRQN_model import Estimator


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])


class Memory:
    def __init__(self, memory_size, min_replay_size, batch_size) -> None:
        self.memory_size = memory_size
        self.min_replay_size = min_replay_size

        self.batch_size = batch_size

        self.memory = []

    def ready(self):
        if len(self.memory) >= self.min_replay_size:
            return True
        return False

    def add(self, experience):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)

        self.memory.append(experience)

    def sample(self):
        return random.sample(self.memory, self.batch_size)


class DRQNAgent(object):
    def __init__(
        self,
        target_update_frequency=1000,
        max_epsilon=1,
        min_epsilon=0.1,
        epsilon_decay_steps=20000,
        gamma=0.99, #discount_factor
        lr = 0.00005,
        memory_size = 10000,
        min_replay_size = 100,
        batch_size=32,
        num_actions=2,
        state_shape=None,
        train_every=1,
        mlp_layers=None,
        lstm_hidden_size=100,
        device=None,
        save_path=None,
        save_every=float('inf'),
    ) -> None:
        self.use_raw = False
        self.memory_size = memory_size
        self.target_update_frequency = target_update_frequency
        self.gamma = gamma
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.train_every = train_every

        self.lr = lr

        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_layers = mlp_layers

        self.state_shape = state_shape

        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # The epsilon decay scheduler
        self.epsilons = np.linspace(min_epsilon, max_epsilon, epsilon_decay_steps)


        self.q_net = Estimator(
            num_actions=self.num_actions,
            learning_rate=self.lr,
            state_shape=self.state_shape,
            mlp_hidden_layer_sizes=self.mlp_layers,
            device=self.device
        )
        self.target_net = Estimator(
            num_actions=self.num_actions,
            lstm_hidden_size=self.lstm_hidden_size,
            learning_rate=self.lr,
            state_shape=self.state_shape,
            mlp_hidden_layer_sizes=self.mlp_layers,
            device=self.device
        )

        self.target_net.load_state_dict(self.q_net.state_dict())


        self.memory = Memory(
            memory_size=memory_size,
            min_replay_size=min_replay_size,
            batch_size=batch_size,
        )

        # Checkpoint saving parameters
        self.save_path = save_path
        self.save_every = save_every
