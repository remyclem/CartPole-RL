
import matplotlib.pyplot as plt
import gym
import numpy as np
import bisect
import math
import os
import random
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.WARN)  # Remove Info logs


def run_episode_random_neural_network(environment, show_renderer=False):
    tf.reset_default_graph()

    nb_neurons_input = 4  # len(obs)
    nb_neurons_hidden_layer = 4
    nb_neurons_output = 1

    X = tf.placeholder(shape=[None, nb_neurons_input], dtype=tf.float32, name="X")

    with tf.name_scope("dnn"):
        initializer_w = tf.contrib.layers.xavier_initializer()
        initializer_b = tf.zeros_initializer()
        hidden = tf.layers.dense(X, nb_neurons_hidden_layer, \
                                 activation=tf.nn.elu, \
                                 kernel_initializer=initializer_w, \
                                 name="hidden")
        output = tf.layers.dense(hidden, nb_neurons_output,
                                 activation=tf.nn.sigmoid,
                                 name="output")  # proba of moving kart to left

    init = tf.global_variables_initializer()

    with tf.Session() as sess\
            :
        init.run()
        obs = environment.reset()
        obs_pretty = obs.reshape(1, nb_neurons_input)
        done = False
        final_score = 0
        while not done:
            if show_renderer:
                environment.render()
            proba_move_to_left = output.eval(feed_dict={X: obs_pretty})
            if random.uniform(0, 1) < proba_move_to_left:
                action = 0  # move to left
            else:
                action = 1
            obs, reward, done, info = environment.step(action)
            if done:
                break
            final_score += reward
    environment.reset()
    environment.close()
    return final_score


def evaluate(environment, nb_episodes=20):
    scores = []
    for i in range(0, nb_episodes):
        episode_score = run_episode_random_neural_network(environment, True)
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
