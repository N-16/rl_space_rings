import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import numpy as np
import gym
import random

import matplotlib.pyplot as plt
from collections import deque


class BasicBuffer:

  def __init__(self, max_size):
      self.max_size = max_size
      self.buffer = deque(maxlen=max_size)

  def push(self, state, action, reward, next_state, done):
      experience = (state, action, np.array([reward]), next_state, done)
      self.buffer.append(experience)

  def sample(self, batch_size):
      state_batch = []
      action_batch = []
      reward_batch = []
      next_state_batch = []
      done_batch = []

      batch = random.sample(self.buffer, batch_size)

      for experience in batch:
          state, action, reward, next_state, done = experience
          state_batch.append(state)
          action_batch.append(action)
          reward_batch.append(reward)
          next_state_batch.append(next_state)
          done_batch.append(done)

      return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

  def __len__(self):
      return len(self.buffer)


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    for episode in range(max_episodes):
        state = env.reset()[0]
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, terminated, truncuated, _ = env.step(action)
            done = terminated or truncuated
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state
    return episode_rewards

class DQN(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals


class DQNAgent:

    def __init__(self, env, learning_rate=3e-4, gamma=0.99, tau=0.01, buffer_size=100000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
	
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
        self.target_model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
        
        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        
    def get_action(self, state, eps=0.20):
        #state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        #print(type(state))
        state = torch.Tensor(state).to(self.device)
        qvals = self.model.forward(state)
        action = torch.argmax(qvals).item()
        
        
        if(np.random.randn() < eps):
            return self.env.action_space.sample()

        return action

    def compute_loss(self, batch):     
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # resize tensors
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        # compute loss
        curr_Q = self.model.forward(states).gather(1, actions)
        next_Q = self.target_model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q
        
        loss = F.mse_loss(curr_Q, expected_Q.detach())
        
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # target network update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


env_id = "CartPole-v1"
MAX_EPISODES = 100000
MAX_STEPS = 500
BATCH_SIZE = 32

env = gym.make(env_id)
agent = DQNAgent(env)
episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)
plt.plot(range(MAX_EPISODES), episode_rewards)
plt.show()