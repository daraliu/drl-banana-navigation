# Report

This repository contains a simple implementation of Deep Q-Network agent and the code required to 
train it for Banana Navigation environment. The goal for the agent is to collect
as many yellow bananas as possible to during an episode while avoiding blue bananas.

The environment is considered solved if the average score of the last 100 episodes is above 13.0.

The DeepMind paper in Nature describing DQN can be found [here](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

## Learning Algorithm

DQN Agent with Experience Replay and Fixed Q targets is used to solve the environment. Details provided below.


### Hyperparameters

The following hyperparameters were used for training.

- `max_t=1000` maximum number of episodes
- `eps_start=1.0` starting value of epsilon
- `eps_end=0.01` minimum value of epsilon
- `eps_decay=0.995` multiplicative factor (per episode) for decreasing epsilon

- `buffer_size=100000` maximum buffer size
- `batch_size=64` size of every training batch
- `gamma=0.99` discount factor
- `tau=1e-3` interpolation parameter for soft network updates
- `learning_rate=5e-4` learning rate
- updated network weights every 4 time steps

### Neural Network Model Architectures

Neural Network with 3 fully connected layers:

- fc1, in:`state_size`, out:64, relu activation
- fc2, in: 64, out:64, relu activation
- fc3, in: 64, out: `action_size`, softmax activation

here `state_size=37`, `action_size=4`.

## Plot of Rewards


Environment was solved in 940 episodes with average score of 13.01.

Scores every 5 episodes are reported in [Navigation.ipynb](https://github.com/daraliu/drl-banana-navigation/blob/master/notebooks/Navigation.ipynb) Jupyter notebook.

![](https://github.com/daraliu/drl-banana-navigation/blob/master/img/baseline_fc_64_64_score.png)


## Ideas for Future Work

To improve agent performance, the following steps should be taken:

- Experiment with more Neural Network architectures, i.e. add convolutional layers to the network.
- Implement Double DQN algorithm. Paper [here](https://arxiv.org/abs/1509.06461).
- Implement Prioritized Experience Replay. Paper [here](https://arxiv.org/abs/1511.05952)
- Implement Dueling DQN. Paper [here](https://arxiv.org/abs/1511.06581).
- Implement multi-step bootstrap targets. Paper [here](https://arxiv.org/abs/1602.01783).
- Implement Distributional DQN. Paper [here](https://arxiv.org/abs/1707.06887).
- Implement Noisy DQN. Paper [here](https://arxiv.org/abs/1706.10295).
- Combine the above DNQ improvements into one - implement Rainbow algorithm. Paper [here](https://arxiv.org/abs/1710.02298).
- Perform hyper parameter turing.
