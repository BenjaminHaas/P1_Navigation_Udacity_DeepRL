#!/usr/bin/env python3

from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from unityagents import UnityEnvironment
import torch

from common import BANANA_FILE
from dqn_agent import Agent

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    # Get environment instance
    env = UnityEnvironment(file_name=BANANA_FILE)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # Reset environment
    env_info = env.reset(train_mode=True)[brain_name]
    # Get initial state, state size and action size
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    # Setup agent
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
   
    # Train!
    max_avg_score = -100000            # max avg score over 100 episodes
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset(train_mode=True)[brain_name].vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0 and np.mean(scores_window) > max_avg_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            # break
            max_avg_score = np.mean(scores_window)

    # Close environment
    env.close()
    return scores

def main():
    # train
    scores = dqn()
    
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    rolling_window = 100
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

if __name__ == "__main__":
    main()
