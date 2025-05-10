# CartPole DQN Agent

This project implements a Deep Q-Network (DQN) agent to solve the CartPole environment from OpenAI Gym. The goal is to balance a pole on a moving cart by applying reinforcement learning techniques.

## Features

- Implementation of a DQN agent using PyTorch.
- Training on the CartPole-v1 environment.
- Replay buffer for experience replay integrated within the agent.
- Adjustable hyperparameters for experimentation.

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- OpenAI Gym

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/cartpole-dqn.git
    cd cartpole-dqn
    ```

2. Train the agent:
    ```bash
    python train.py
    ```

## File Structure

- `train.py`: Script to train the DQN agent.
- `dqn_agent.py`: Implementation of the DQN agent.
- `README.md`: Project documentation.

## References

- [OpenAI Gym](https://www.gymlibrary.dev/)
- [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
