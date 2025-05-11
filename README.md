# CartPole DQN Agent

This project implements a Deep Q-Network (DQN) agent to solve the CartPole environment from OpenAI Gym. The goal is to balance a pole on a moving cart by applying reinforcement learning techniques.

## Features

- Implementation of a DQN agent using PyTorch.
- Training on the CartPole-v1 environment.
- Replay buffer for experience replay integrated within the agent.
- Adjustable hyperparameters for experimentation.
- Evaluation of the trained model with visualization.
- GIF generation to visualize the agent's performance.

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- OpenAI Gym
- Matplotlib
- Imageio
- Pygame

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/rahulbedjavalge/cartpole-dqn.git
    cd cartpole-dqn
    ```

2. Train the agent:
    ```bash
    python train.py
    ```

3. Evaluate the trained model and generate a GIF:
    ```bash
    python evaluate.py
    ```

## File Structure

- `train.py`: Script to train the DQN agent.
- `evaluate.py`: Script to evaluate the trained model and generate a GIF of the agent's performance.
- `dqn_agent.py`: Implementation of the DQN agent.
- `dqn_cartpole.pth`: Saved model weights after training.
- `training.gif`: Visualization of the agent's performance.
- `requirements.txt`: List of dependencies for the project.
- `README.md`: Project documentation.

## References

- [OpenAI Gym](https://www.gymlibrary.dev/)
- [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
