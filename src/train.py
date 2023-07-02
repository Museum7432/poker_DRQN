import rlcard
import sys
import torch
import numpy as np
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils.utils import tournament
from rlcard.utils import (
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
from agents.DRQN_agent import DRQNAgent


# Make environments to train and evaluate models
env = rlcard.make('limit-holdem')
eval_env = rlcard.make('limit-holdem')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Number of actions:", env.num_actions)
print("Number of players:", env.num_players)
print("Shape of state:", env.state_shape)
print("Shape of action:", env.action_shape)


# initialize DRQN agents
drqn_agents = []
for i in range(2):
    drqn_agents.append(DRQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        lstm_hidden_size=150,
        mlp_layers=[128,256],
        
    ))


env.set_agents(drqn_agents)



eval_every = 100
eval_num = 1000
episode_num = 300000



for episode in range(episode_num):

    # reset hidden state of recurrent agents
    for i in range(2):
        drqn_agents[i].reset_hidden_and_cell()

    # get transitions by playing an episode in env
    trajectories, payoffs = env.run(is_training=True)
    trajectories = reorganize(trajectories, payoffs)



    for i in range(2):
        drqn_agents[i].feed(trajectories[i])




