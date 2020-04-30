#!/usr/bin/env python3

from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import numpy as np

from common import BANANA_FILE
from dqn_agent import Agent

# Get environment instance
env = UnityEnvironment(file_name=BANANA_FILE)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=False)[brain_name]
# get initial state
state = env_info.vector_observations[0]
# get state size and actions size
state_size = len(state)
action_size = brain.vector_action_space_size
# create Agent instance
agent = Agent(state_size=state_size, action_size=action_size, seed=0, weights='weights.pth')
# test the Agent
scores = []
for i_episode in range(100):
    score = 0
    done = False
    while not done:
        action = agent.act(state)                           # select an action
        env_info = env.step(action)[brain_name]             # send the action to the environment
        state = env_info.vector_observations[0]             # update the state
        reward = env_info.rewards[0]                        # get the reward
        done = env_info.local_done[0]                       # see if episode has finished
        score += reward                                     # update the score
        if done:                                            # exit loop if episode finished
            break
    env_info = env.reset(train_mode=False)[brain_name]      # reset the environment
    state = env_info.vector_observations[0]
    print("Episode: {} Score: {}".format(i_episode, score))
    scores.append(score)
# close the environment
env.close()
# print average score
print("Average Score: {}".format(np.mean(scores)))
# plot the scores
fig = plt.figure()
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()