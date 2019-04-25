# coding: utf-8

import matplotlib
import matplotlib.pyplot as plt
import gym
import numpy as np
import bisect
import math
import os
import random


def human_decision(angle):
    if angle < 0:
        action = 0  # move Cart to left
    else:
        action = 1  # move Cart to right
    return action

def run_episode_human_crafted_agent(environment, show_renderer=False):
    obs = environment.reset()
    done = False
    final_score = 0
    while not done:
        if show_renderer:
            environment.render()
        position, velocity, angle, angular_velocity = obs
        action = human_decision(angle)
        obs, reward, done, info = environment.step(action)
        final_score += reward
    environment.reset()
    environment.close()
    return final_score

def evaluate(environment, nb_episodes=20):
    scores = []
    for i in range(0, nb_episodes):
        episode_score = run_episode_human_crafted_agent(environment, True)
        scores.append(episode_score)

    mean_reward = np.mean(scores)
    plt.plot(scores)
    plt.title("Mean reward: {}".format(mean_reward))
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.show()


if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    evaluate(env)
