import gym
import torch
import numpy as np
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import imageio

# Load the environment
env = gym.make("CartPole-v1", render_mode="rgb_array")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Load the trained model
SAVE_PATH = 'C:/Users/it/cartpole-dqn/dqn_cartpole.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNAgent(state_size, action_size, device)
agent.qnetwork.load_state_dict(torch.load(SAVE_PATH))
agent.qnetwork.eval()

# Evaluate the model
eval_episodes = 500  # Limit to 5 episodes
total_rewards = []
frames = []

for episode in range(eval_episodes):
    state = env.reset()[0]  # gym 0.26+ returns (obs, info)
    total_reward = 0

    while True:
        # Render the environment and save every 5th frame for the GIF
        if len(frames) % 5 == 0:
            frame = env.render()
            frames.append(frame)

        # Select action using the trained model
        action = agent.select_action(state, epsilon=0.0)  # Greedy policy
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = next_state
        total_reward += reward

        if done:
            break

    total_rewards.append(total_reward)
    print(f"Evaluation Episode {episode + 1}, Total Reward: {total_reward}")

# Save the GIF with adjusted FPS
gif_path = "C:/Users/it/cartpole-dqn/training.gif"
imageio.mimsave(gif_path, frames, fps=20)  # Adjust FPS for smoother playback
print(f"Training visualization saved as GIF at {gif_path}")

# Plot the rewards
plt.plot(total_rewards)
plt.title("Evaluation Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

env.close()
