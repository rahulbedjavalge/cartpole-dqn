import gym
import torch 
import numpy as np 
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import IPython.display as clear_output
import os 

# Add the following code to stop GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Hyperparameters
EPISODES = 500
MAX_STEPS = 500
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
SAVE_PATH = 'C:/Users/it/cartpole-dqn/dqn_cartpole.pth'

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create envirnment 
env =gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]  # should be 4
action_size = env.action_space.n # should be 2

# create agent 
agent = DQNAgent(state_size, action_size, device)
epsilon = EPS_START

# Add imports for plotting
import matplotlib.pyplot as plt

# Function to plot training rewards
def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Total Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards Over Episodes')
    plt.legend()
    plt.show()

# Modify training loop to collect rewards
all_rewards = []

for episode in range(EPISODES):
    state = env.reset()[0]  # gym 0.26+ returns (obs, info)
    total_reward = 0

    for t in range(MAX_STEPS):
        action = agent.select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)  # Fixed typo in variable names
        done = terminated or truncated

        # Replace `np.bool8` with `bool` to fix compatibility issues
        if not isinstance(terminated, (bool, bool)):
            terminated = bool(terminated)
        if not isinstance(truncated, (bool, bool)):
            truncated = bool(truncated)

        agent.store(state, action, reward, next_state, done)
        agent.train_step()  # Fixed typo in method call

        state = next_state
        total_reward += reward

        if done:
            break

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    all_rewards.append(total_reward)

    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    # Save the model
    torch.save(agent.qnetwork.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

# Plot rewards after training
plot_rewards(all_rewards)

# Load the saved model for evaluation
def evaluate_model():
    agent.qnetwork.load_state_dict(torch.load(SAVE_PATH))
    agent.qnetwork.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {SAVE_PATH}")

    eval_episodes = 10
    total_rewards = []

    for episode in range(eval_episodes):
        state = env.reset()[0]  # gym 0.26+ returns (obs, info)
        total_reward = 0

        while True:
            action = agent.select_action(state, epsilon=0.0)  # Use greedy policy for evaluation
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

            if done:
                break

        total_rewards.append(total_reward)
        print(f"Evaluation Episode {episode + 1}, Total Reward: {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {eval_episodes} episodes: {avg_reward}")

# Uncomment the following line to evaluate the model after training
# evaluate_model()
