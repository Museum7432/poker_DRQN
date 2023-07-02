import os
import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
# for testing
from torchinfo import summary

from .models.DRQN_model import Estimator


Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done", "legal_actions"]
)


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
        gamma=0.99,  # discount_factor
        lr=0.00005,
        memory_size=10000,
        min_replay_size=100,
        batch_size=32,
        num_actions=2,
        state_shape=None,
        train_every=1,
        mlp_layers=None,
        lstm_hidden_size=100,
        device=None,
        save_path=None,
        save_every=float("inf"),
    ) -> None:
        self.use_raw = False
        self.memory_size = memory_size
        self.target_update_frequency = target_update_frequency
        self.gamma = gamma
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.train_every = train_every

        self.min_replay_size = min_replay_size

        self.lr = lr

        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_layers = mlp_layers

        self.state_shape = state_shape

        self.lstm_input_size = state_shape[0]

        # Torch device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            lstm_hidden_size=self.lstm_hidden_size,
            learning_rate=self.lr,
            state_shape=self.state_shape,
            mlp_hidden_layer_sizes=self.mlp_layers,
            device=self.device,
        )

        print(self.mlp_layers)

        self.target_net = Estimator(
            num_actions=self.num_actions,
            lstm_hidden_size=self.lstm_hidden_size,
            learning_rate=self.lr,
            state_shape=self.state_shape,
            mlp_hidden_layer_sizes=self.mlp_layers,
            device=self.device,
        )

        # print(self.q_net.qnet.state_dict())
        self.target_net.qnet.load_state_dict(self.q_net.qnet.state_dict())


        # summary(self.target_net.qnet, input_size=(1,72))

        # summary(self.q_net.qnet, input_size=(1,72))

        # exit()
        

        self.memory = Memory(
            memory_size=memory_size,
            min_replay_size=min_replay_size,
            batch_size=batch_size,
        )

        # Checkpoint saving parameters
        self.save_path = save_path
        self.save_every = save_every

    def feed(self, seq_of_transition):
        if len(seq_of_transition) == 0:
            return
        seq = []

        # print(seq_of_transition)

        for ts in seq_of_transition:
            
            (state, action, reward, next_state, done) = tuple(ts)
            
            seq.append(
                Transition(
                    state=state["obs"],
                    action=action,
                    reward=reward,
                    next_state=next_state["obs"],
                    legal_actions=list(next_state["legal_actions"].keys()),
                    done=done
                )
            )

        self.memory.add(seq)
        self.total_t += 1

        if (
            self.memory.ready()
            and (self.total_t - self.min_replay_size) % self.train_every == 0
        ):
            self.train()

    def step(self, state):
        """Predict the action for genrating training data but
            have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        """
        q_values = self.predict(state)

        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]

        legal_actions = list(state["legal_actions"].keys())

        probs = np.ones(len(legal_actions), dtype=float) * epsilon / len(legal_actions)

        best_action_idx = legal_actions.index(np.argmax(q_values))

        probs[best_action_idx] += 1.0 - epsilon
        action_idx = np.random.choice(np.arange(len(probs)), p=probs)

        return legal_actions[action_idx]
    
    def eval_step(self, state):
        ''' Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
            info (dict): A dictionary containing information
        '''
        q_values = self.predict(state)
        best_action = np.argmax(q_values)

        info = {}
        info['values'] = {state['raw_legal_actions'][i]: float(q_values[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}

        return best_action, info

    def predict(self, state):
        legal_actions = list(state["legal_actions"].keys())

        state = state["obs"]
        state = torch.tensor(state, dtype=torch.float32).view(-1,len(state)).to(self.device)
        q_values = self.q_net.predict_nograd(state)[0]

        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)

        

        for a in legal_actions:
            masked_q_values[a] = q_values[a]


        return masked_q_values

    def train(self):
        """Train the network

        Returns:
            loss (float): The loss of the current batch.
        """
        # state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample()
        sequences = self.memory.sample()

        # calculate the q value of each state in each seq

        target_q_values_per_seq = []

        for seq in sequences:
            self.target_net.qnet.reset_hidden_and_cell()
            next_states = np.array([t[3] for t in seq])
            rewards = [t[2] for t in seq]
            done = [t[4] for t in seq]


            next_states = torch.FloatTensor(next_states).view(-1,1,self.lstm_input_size).to(self.device)

            

            q_values_next_target = self.target_net.predict_nograd(next_states)

            # Compute targets using the formulation sample = r + gamma * max q(s',a')
            max_target_q_values = q_values_next_target.max(axis=-1)

            q_values_target = []

            for i in range(len(q_values_next_target)):
                if done[i]:
                    q_values_target.append(rewards[i])
                else:
                    q_values_target.append(rewards[i] + self.gamma*max_target_q_values[i][0])
            

            target_q_values_per_seq.append(np.asarray(q_values_target,dtype=np.float32))
    


        loss = self.q_net.update(sequences, target_q_values_per_seq)
        print("\rINFO - Step {}, rl-loss: {}".format(self.total_t, loss), end="")

        # Update the target estimator
        if self.train_t % self.target_update_frequency == 0:
            # self.target_net = deepcopy(self.q_net)
            self.target_net.qnet.load_state_dict(self.q_net.qnet.state_dict())
            print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1

        if self.save_path and self.train_t % self.save_every == 0:
            # To preserve every checkpoint separately,
            # add another argument to the function call parameterized by self.train_t
            self.save_checkpoint(self.save_path)
            print("\nINFO - Saved model checkpoint.")
    
    def set_device(self, device):
        self.device = device
        self.q_net.device = device
        self.target_net.device = device

    def reset_hidden_and_cell(self):
        self.q_net.qnet.reset_hidden_and_cell()
        self.target_net.qnet.reset_hidden_and_cell()
