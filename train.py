import gym 
import torch 
import numpy as np 
from dqn_agent import DQNAgent

# Hyperparameters
EPISODES = 500
MAX_STEPS = 500
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
SAVE_PATH = 'dqn_cartpole.pth'

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create envirnment 
env =gym.make("cartpole-v1")
state_size =env.observtion_space.shpae[0]  # should be 4
action_size = env.action_space.n # should be 2

# create agent 
agent = DQNAgent(state_size, action_size, device)
epsilon = eps_start

for episod in range(EPISODES):
    state = env.reset()[0] #gym 0.26+ returns (obs, info)
    total_reward = 0
    
    for t in range(MAX_STEPS):
        action = agent.select_action(state, epsilon)
        next-state, reward, terminated, tuncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store(state, action, reward, next_state, done)
        agent,train_step()
        
        state = next_state
        total_reward +=reward
        
        if done:
            break
        
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    
    print(f"Episode {episod + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
    
    # save the model 
    torch.save(agent.q_network.state_dict(), SAVE_PATH)
    print(f"model saved to {SAVE_PATH}")
    env.close()
    