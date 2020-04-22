import json
import logging
import pathlib
from collections import deque

import numpy as np
import torch
import typing
import unityagents
import pandas as pd

from banana_nav import agents
from banana_nav import config as cfg
from banana_nav import path_util


def training(
        env: unityagents.UnityEnvironment,
        output_dir: typing.Union[pathlib.Path, str],
        agent_type: str = "dqn",
        n_episodes: int = 2000,
        mean_score_threshold: float = 13.0,
        max_t: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.995,
        agent_seed=0,
        logging_freq: int = 10):
    """
    Train agent for Unity Banana Navigation environment and save results.

    Train a deep reinforcement learning agent to pick up yellow bananas and
    avoid blue bananas in Unity Banana Navigation Environment and save
    results (training scores, agent neural network model weights, metadata with hyper-parameters)
    to provided output directory.

    Parameters
    ----------
    env
        Unity environment
    output_dir
        Path to output results output directory (scores, weights, metadata)
    agent_type: one of {dqn},
        A type of agent to train from the available ones
    n_episodes
        Maximum number of episodes
    mean_score_threshold
        Threshold of mean last 100 weights to stop training and save results
    max_t:
        Maximum number of time steps per episode
    eps_start
        Starting value of epsilon, for epsilon-greedy action selection
    eps_end
        Minimum value of epsilon
    eps_decay
        Multiplicative factor (per episode) for decreasing epsilon
    agent_seed
        Random seed for agent epsilon-greedy policy
    logging_freq
        Logging frequency

    """
    logger = logging.getLogger(__name__)

    output_dir = pathlib.Path(output_dir)

    logger.info(f"Ensuring output directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    path_weights = path_util.mk_path_weights(output_dir)
    path_scores = path_util.mk_path_scores(output_dir)
    path_metadata = path_util.mk_path_metadata(output_dir)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size

    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    state_size = len(state)

    agent = agents.DQNAgent(state_size=state_size, action_size=action_size, seed=agent_seed)

    scores = train_agent(
        env,
        agent,
        n_episodes,
        mean_score_threshold,
        max_t,
        eps_start,
        eps_end,
        eps_decay,
        logging_freq)

    logger.info(f'Saving network model weights to {str(path_weights)}')
    torch.save(agent.qnetwork_local.state_dict(), str(path_weights))
    logger.info(f'Model weights saved successfully!')

    logger.info(f'Saving training scores to {str(path_scores)}')
    scores_df = pd.DataFrame.from_records(
        enumerate(scores, start=1),
        columns=(cfg.SCORE_COLNAME_X, cfg.SCORE_COLNAME_Y))
    logger.info(f'Training scores saved successfully!')

    scores_df.to_csv(path_scores, index=False)

    logger.info(f'Saving training metadata to {str(path_metadata)}')
    metadata = {
        "agent_type": agent_type,
        "agent": agent.metadata,
        "mean_score_threshold": mean_score_threshold,
        "max_t": max_t,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "eps_decay": eps_decay,
    }
    with open(path_metadata, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f'Training metadata saved successfully!')


def train_agent(
        env: unityagents.UnityEnvironment,
        agent: agents.DQNAgent,
        n_episodes: int = 2000,
        mean_score_threshold: float = 13.0,
        max_t: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.995,
        logging_freq: int = 10) -> typing.List[float]:
    """
    Train agent for Unity Banana Navigation environment and return scores.

    Train a deep reinforcement learning agent to pick up yellow bananas and
    avoid blue bananas in Unity Banana Navigation Environment and save
    results (training scores, agent neural network model weights, metadata with hyper-parameters)
    to provided output directory.

    Parameters
    ----------
    env
        Unity environment
    agent
        And instance of Deep Reinforcement Learning Agent from banana_nav.agents module
    n_episodes
        Maximum number of episodes
    mean_score_threshold
        Threshold of mean last 100 weights to stop training and save results
    max_t:
        Maximum number of time steps per episode
    eps_start
        Starting value of epsilon, for epsilon-greedy action selection
    eps_end
        Minimum value of epsilon
    eps_decay
        Multiplicative factor (per episode) for decreasing epsilon
    logging_freq
        Logging frequency

    """

    logger = logging.getLogger(__name__)

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(1, n_episodes+1):
        brain_name = env.brain_names[0]
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done = env_step(env, action)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)

        eps = max(eps_end, eps_decay * eps)
        if i_episode % logging_freq == 0:
            logger.info(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

        if np.mean(scores_window) >= mean_score_threshold:
            logger.info(
                f'\nEnvironment solved in {i_episode-100:d} episodes!'
                f'\tAverage Score: {np.mean(scores_window):.2f}')
            break

    return scores


def env_step(env: unityagents.UnityEnvironment, action: int) -> 'EnvStep':
    """
    Return next_state, reward, done tuple from environment given an action

    Parameters
    ----------
    env
        Unity Environment
    action
        Agent action

    Returns
    -------
    EnvStep
        next_state, reward, done tuple

    """
    brain_name = env.brain_names[0]
    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    return EnvStep(next_state, reward, done)


def demo(
        env: unityagents.UnityEnvironment,
        path_weights: typing.Optional[pathlib.Path] = None
) -> float:
    """
    Run a demo on the environment

    Parameters
    ----------
    env
        Unity Environment
    path_weights
        If provided, agent neural network weights are loaded from path,
        Random Agent is used otherwise

    Returns
    -------
    float
        final score

    """
    if path_weights is not None:
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        action_size = brain.vector_action_space_size
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        state_size = len(state)

        agent = agents.DQNAgent(state_size=state_size, action_size=action_size)
        agent.qnetwork_local.load_state_dict(torch.load(path_weights))

        return demo_trained(env, agent)
    else:
        return demo_random(env)


def demo_trained(env: unityagents.UnityEnvironment, agent) -> float:
    """
    Run a demo of a trained agent

    Parameters
    ----------
    env
        Unity Environment
    agent
        trained agent

    Returns
    -------
    float
        final score

    """
    brain_name = env.brain_names[0]

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state

    score = 0  # initialize the score
    while True:
        action = agent.act(state)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break

    return score


def demo_random(env: unityagents.UnityEnvironment) -> float:
    """
    Run a demo of a Random Agent

    Parameters
    ----------
    env
        Unity Environment

    Returns
    -------
    float
        final score

    """
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=False)[brain_name]

    score = 0  # initialize the score
    while True:
        action = np.random.randint(action_size)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        if done:  # exit loop if episode finished
            break

    return score


class EnvStep(typing.NamedTuple):
    next_state: np.array
    reward: float
    done: bool
