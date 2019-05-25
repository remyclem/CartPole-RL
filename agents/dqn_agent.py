# coding: utf-8

import matplotlib.pyplot as plt
import gym
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
import random

tf.logging.set_verbosity(tf.logging.WARN)  # Remove Info logs

# Here this is just a single DQN with no experience replay

def train_dqn_neural_network(environment,
                             save_folder):
    tf.reset_default_graph()

    nb_neurons_input = environment.observation_space.shape[0]
    nb_neurons_hidden_layer = 4
    nb_neurons_output = 2  # q_value for going left, q_value for going right
    learning_rate = 0.001

    X = tf.placeholder(tf.float32, shape=[None, nb_neurons_input], name="X")
    target_q_value = tf.placeholder(tf.float32, shape=[None, nb_neurons_output], name="target_q_value")

    with tf.name_scope("dnn"):
        initializer_w = tf.contrib.layers.xavier_initializer()
        initializer_b = tf.zeros_initializer()

        hidden_1 = tf.layers.dense(X, nb_neurons_hidden_layer, \
                                   activation=tf.nn.relu, \
                                   kernel_initializer=initializer_w, \
                                   bias_initializer=initializer_b, \
                                   name="hidden_1")
        hidden_2 = tf.layers.dense(hidden_1, nb_neurons_hidden_layer, \
                                   activation=tf.nn.relu, \
                                   kernel_initializer=initializer_w, \
                                   bias_initializer=initializer_b, \
                                   name="hidden_2")
        estimated_q_value = tf.layers.dense(hidden_2, nb_neurons_output,
                                            name="estimated_q_value")

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.pow(target_q_value - estimated_q_value, 2), name="loss")
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        #init.run()
        saver = tf.train.import_meta_graph(os.path.join(save_folder, "dqn_neural_net.ckpt.meta"))
        saver.restore(sess, tf.train.latest_checkpoint(save_folder))

        epsilon_exploration = 1
        exploration_decay_rate = 0.99
        min_exploration_rate = 0.001
        discount_rate = 0.95
        nb_training_episode = 2000

        # A virer
        scores = []


        for episode in tqdm(range(nb_training_episode)):
            #print("################## Episode: " + str(episode))
            done = False
            obs = environment.reset()
            step = 0

            episode_memory = []  # state, action, reward, q_value

            while not done:
                obs_pretty = obs.reshape(1, obs.shape[0])
                estimated_q_values_current_state = estimated_q_value.eval(feed_dict={X: obs_pretty})

                if random.random() <= epsilon_exploration:
                    selected_action = random.randint(0, 1)
                else:
                    selected_action = np.argmax(estimated_q_values_current_state)
                new_obs, reward, done, info = env.step(selected_action)
                new_obs_pretty = new_obs.reshape(1, obs.shape[0])

                # print("---- step: " + str(step))
                # print("selected_action: " + str(selected_action))
                # print("done: " + str(done))
                # print("estimated_q_values_current_state: " + str(estimated_q_values_current_state))








                if done and step < 499:
                    reward = -10
                    max_q_value_next_state = 0
                else:
                    reward = 0.1
                    estimated_q_values_next_state = estimated_q_value.eval(feed_dict={X: new_obs_pretty})
                    max_q_value_next_state = np.amax(estimated_q_values_next_state)

                target_q_for_selected_action = reward + discount_rate * max_q_value_next_state
                target_q_values_current_state = estimated_q_values_current_state
                target_q_values_current_state[target_q_values_current_state < -10] = -10  # Fasten the learning at start
                target_q_values_current_state[0][selected_action] = target_q_for_selected_action
                # print("target_q_for_selected_action: " + str(target_q_for_selected_action))
                # print("target_q_values_current_state: " + str(target_q_values_current_state))

                sess.run(training_op, feed_dict={X: obs_pretty, target_q_value: target_q_values_current_state})

                obs = new_obs
                step += 1
                epsilon_exploration = max(min_exploration_rate, epsilon_exploration * exploration_decay_rate)

            # A virer
            scores.append(step)


        # A virer
        plt.plot(scores)



        tf_log_file = os.path.join(save_folder, "dqn_neural_net.ckpt")
        saver.save(sess, tf_log_file)

def run_episode_dqn_neural_network(environment,
                                   save_folder,
                                   show_renderer=False):
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(save_folder, "dqn_neural_net.ckpt.meta"))
        saver.restore(sess, tf.train.latest_checkpoint(save_folder))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        estimated_q_value = graph.get_tensor_by_name("dnn/estimated_q_value/BiasAdd:0")

        obs = environment.reset()
        done = False
        final_score = 0
        #print("######")
        while not done:
            if show_renderer:
                environment.render()
            obs_pretty = obs.reshape(1, obs.shape[0])
            estimated_q_values = estimated_q_value.eval(feed_dict={X: obs_pretty})
            #print("estimated_q_values: " + str(estimated_q_values))
            q_value_go_left = estimated_q_values[0][0]
            q_value_go_right = estimated_q_values[0][1]
            if q_value_go_right > q_value_go_left:
                action = 1
            else:
                action = 0
            #print("action taken; " + str(action))
            obs, reward, done, info = environment.step(action)

            final_score += reward
    environment.close()
    return final_score

def evaluate(environment, save_folder, nb_episodes=20, show_renderer=True):
    scores = []
    for i in range(0, nb_episodes):
        episode_score = run_episode_dqn_neural_network(environment, save_folder, show_renderer)
        scores.append(episode_score)

    mean_reward = np.mean(scores)
    plt.plot(scores)
    plt.title("Mean reward: {}".format(mean_reward))
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.show()
    return mean_reward


if __name__ == '__main__':
    env = gym.make("CartPole-v1")

    save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "save", "dqn_neural_network")
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # Training
    train_dqn_neural_network(env, save_folder)

    # Evaluation of the imitation neural network
    # Hopefully it should perform as well as the human crafted policy
    mean_reward = evaluate(env, save_folder, show_renderer=False)
    print("Mean reward per episode: " + str(mean_reward))
