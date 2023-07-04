import rlcard
import os.path
import torch
from rlcard.agents import RandomAgent
from rlcard.utils.utils import tournament
from rlcard.utils import (
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
from rlcard import models

from agents.DRQN_agent import DRQNAgent
import signal
from torch.utils.tensorboard import SummaryWriter


# Make environments to train and evaluate models
env = rlcard.make("limit-holdem")
eval_env = rlcard.make("limit-holdem")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Number of actions:", env.num_actions)
print("Number of players:", env.num_players)
print("Shape of state:", env.state_shape)
print("Shape of action:", env.action_shape)
print("device:", device)


target_update_frequency = 700
max_epsilon = 1
min_epsilon = 0.1
epsilon_decay_steps = 20000
gamma = 0.99  # discount_factor
lr = 0.00005
memory_size = 100000
min_replay_size = 200
batch_size = 64
num_actions = env.num_actions
state_shape = env.state_shape[0]
train_every = 1
mlp_layers = [256, 512]
lstm_hidden_size = 128
save_path = "saves"
save_every = 1000


eval_every = 500
eval_num = 1000
episode_num = 300000

update_every = 150
# initialize DRQN agents

drqn_agent = DRQNAgent(
    target_update_frequency=target_update_frequency,
    max_epsilon=max_epsilon,
    min_epsilon=min_epsilon,
    epsilon_decay_steps=epsilon_decay_steps,
    gamma=gamma,  # discount_factor
    lr=lr,
    memory_size=memory_size,
    min_replay_size=min_replay_size,
    batch_size=batch_size,
    num_actions=num_actions,
    state_shape=state_shape,
    train_every=train_every,
    mlp_layers=mlp_layers,
    lstm_hidden_size=lstm_hidden_size,
    save_path=save_path,
    save_every=save_every,
    device=device,
)

train_target = DRQNAgent(
    target_update_frequency=target_update_frequency,
    max_epsilon=max_epsilon,
    min_epsilon=min_epsilon,
    epsilon_decay_steps=epsilon_decay_steps,
    gamma=gamma,  # discount_factor
    lr=lr,
    memory_size=memory_size,
    min_replay_size=min_replay_size,
    batch_size=batch_size,
    num_actions=num_actions,
    state_shape=state_shape,
    train_every=train_every,
    mlp_layers=mlp_layers,
    lstm_hidden_size=lstm_hidden_size,
    save_path=save_path,
    save_every=save_every,
    device=device,
)

pretrained_eval = models.load("limit-holdem-rule-v1").agents[0]

if os.path.isfile(save_path + "/checkpoint_drqn.pt"):
    drqn_agent.from_checkpoint(torch.load(save_path + "/checkpoint_drqn.pt"))
    train_target.from_checkpoint(drqn_agent.checkpoint_attributes())
    drqn_agent.reset_hidden_and_cell()
    train_target.reset_hidden_and_cell()

random_agent = RandomAgent(num_actions=eval_env.num_actions)

eval_env.set_agents([drqn_agent, pretrained_eval])

env.set_agents([drqn_agent, train_target])


logger = SummaryWriter("logs/drqn_drqn_agent")


def handler(signum, frame):
    print("\nexit")
    drqn_agent.save_checkpoint(filename="last.pt")
    exit()

signal.signal(signal.SIGINT, handler)


for episode in range(episode_num):
    if episode % update_every == 0 and drqn_agent.memory.ready():
        print("\ncopy model parameters to target agent")
        train_target.q_net.from_checkpoint(drqn_agent.q_net.checkpoint_attributes())
    # reset hidden state of recurrent agents
    drqn_agent.reset_hidden_and_cell()
    train_target.reset_hidden_and_cell()

    # get transitions by playing an episode in env
    trajectories, payoffs = env.run(is_training=True)
    trajectories = reorganize(trajectories, payoffs)

    drqn_agent.feed(trajectories[0])

    if episode % eval_every == 0:
        score = 0
        for i in range(eval_num):
            drqn_agent.reset_hidden_and_cell()

            score += tournament(eval_env, 1)[0]
        logger.add_scalar("reward vs. random agent", score / eval_num, episode)
