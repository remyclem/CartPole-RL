# coding: utf-8

import matplotlib.pyplot as plt
import gym
import numpy as np


def run_episode_random_agent(environment, show_renderer=False):
    obs = environment.reset()
    done = False
    final_score = 0
    while not done:
        if show_renderer:
            environment.render()
        action = environment.action_space.sample()
        obs, reward, done, info = environment.step(action)
        final_score += reward
    environment.reset()
    environment.close()
    return final_score

def evaluate(environment, nb_episodes=20):
    scores = []
    for i in range(0, nb_episodes):
        episode_score = run_episode_random_agent(environment, True)
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