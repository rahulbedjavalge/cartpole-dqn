import torch 
import torch.nn as nn
import random
import numpy as np
from collections import deque
import torch.optim as optim


# Q network: maps state -> Q - values for each action 
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size) # first hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, action_size) # output layer Q values 
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class ReplayBuffer:
    def __init__(self, capacity = 10000):
        self.buffer = deque(maxlen=capacity) #store experiences up to a fixed size
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state,action, reward, next_state, done))
        
    def sample(self, batch_size = 64):
        batch = random.sample(self.buffer,batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )
        
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, state_size, action_size, device, gamma=0.99, lr=1e-3,batch_size=64, buffer_capacity=10000, target_update_freq=10):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        # Q netork and tahget network 
        self.qnetwork = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.qnetwork.state_dict())  #says weights of target network to q network
        
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)
        
        #Experience replay
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size

        #other hyperparameters
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.steps = 0 # track steps for updateing networks 
        
    def select_action(self, state, epsilon,):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1) # explore: random action
        state =torch.tensor(state, dtype=torch.float32).unsqueeze(0).to (self.device)
        with torch.no_grad():
            q_values = self.qnetwork(state)
        return q_values.argmax().item() # exploit : best known action
    
    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return # not enoght experiences yet 
        
        #sample a batch
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        states = state.to(self.device)
        actions = action.to(self.device)
        rewards = reward.to(self.device)
        next_states = next_state.to(self.device)
        dones = done.to(self.device)

        # Compute q_values and next_q_values
        q_values = self.qnetwork(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]

        # Compute targets
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.functional.mse_loss(q_values, targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.qnetwork.state_dict())
