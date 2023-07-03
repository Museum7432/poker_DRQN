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

from torch.utils.tensorboard import SummaryWriter

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
        batch_size=25,
        min_replay_size=200,
        save_every=500
    ))

drqn_agents[0].from_checkpoint(torch.load("./saves/checkpoint_drqn.pt"))
random_agent = RandomAgent(num_actions=eval_env.num_actions)
env.set_agents(drqn_agents)
eval_env.set_agents([drqn_agents[0], random_agent])

eval_every = 100
eval_num = 1000
episode_num = 300000

update_every = 30

logger = SummaryWriter('logs/drqn_drqn_agent')


for episode in range(episode_num):

    # reset hidden state of recurrent agents
    for i in range(2):
        drqn_agents[i].reset_hidden_and_cell()

    # get transitions by playing an episode in env
    trajectories, payoffs = env.run(is_training=True)
    trajectories = reorganize(trajectories, payoffs)

    drqn_agents[0].feed(trajectories[0])
    
    if episode % update_every == 0:
        drqn_agents[1].q_net.from_checkpoint(drqn_agents[0].q_net.checkpoint_attributes())

    if episode % eval_every == 0:
        score = 0
        for i in range(eval_num):
            for j in range(2):
                drqn_agents[j].reset_hidden_and_cell()
            score += tournament(eval_env, 1)[0]
        logger.add_scalar('reward vs. random agent', score / eval_num, episode)
    



