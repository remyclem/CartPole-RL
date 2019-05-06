# coding: utf-8

import matplotlib.pyplot as plt
import gym
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.WARN)  # Remove Info logs

# A2C stands for Advantage Actor Critic

def train_policy_gradient_reinforced_neural_network(environment,
                                                    save_folder,
                                                    nb_training_episodes=5000):
    tf.reset_default_graph()

    nb_neurons_input = 4  # len(obs)

    # Actor
    nb_neurons_hidden_layer_actor = 4
    nb_neurons_output_actor = 1
    learning_rate_actor = 0.01

    # Critic
    nb_neurons_hidden_layer_critic = 4
    nb_neurons_output_critic = 1
    learning_rate_critic = 0.00001

    X_placeholder = tf.placeholder(tf.float32, shape=[None, nb_neurons_input], name="X")
    target_value_placeholder = tf.placeholder(tf.float32)

    with tf.name_scope("actor"):
        initializer_w = tf.contrib.layers.xavier_initializer()
        initializer_b = tf.zeros_initializer()

        hidden = tf.layers.dense(X_placeholder, nb_neurons_hidden_layer_actor, \
                                 activation=tf.nn.elu, \
                                 kernel_initializer=initializer_w, \
                                 bias_initializer=initializer_b, \
                                 name="hidden_actor")
        logits = tf.layers.dense(hidden, nb_neurons_output_actor,
                                 name="logits")
        estimated_action_value = tf.nn.sigmoid(logits, name="estimated_action_value")  # 0 => left, 1 => right
        p_left_and_right = tf.concat(axis=1, values=[1 - estimated_action_value, estimated_action_value])
        tentative_action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

    with tf.name_scope("critic"):
        initializer_w = tf.contrib.layers.xavier_initializer()
        initializer_b = tf.zeros_initializer()

        hidden = tf.layers.dense(X_placeholder, nb_neurons_hidden_layer_critic, \
                                 activation=tf.nn.elu, \
                                 kernel_initializer=initializer_w, \
                                 bias_initializer=initializer_b, \
                                 name="hidden_critic")
        value_function = tf.layers.dense(hidden, nb_neurons_output_critic,
                                 name="value_function")

    # Loss for the actor
    #loss_actor =
    #optimizer_actor = tf.train.AdamOptimizer(learning_rate_actor, name='actor_optimizer')
    #training_op_actor = optimizer_actor.minimize(loss_actor)

    # Loss for the critic
    loss_critic = tf.reduce_mean(tf.pow(tf.squeeze(value_function) - target_value_placeholder, 2))
    optimizer_critic = tf.train.AdamOptimizer(learning_rate_critic, name='critic_optimizer')
    training_op_critic = optimizer_critic.minimize(loss_critic)

    # TODO: scaling
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    save_iterations = 10
    discount_rate = 0.95
    tf_log_file = os.path.join(save_folder, "a2c_neural_net.ckpt")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init.run()
        for episode in tqdm(range(nb_training_episodes)):
            step = 0
            done = False
            episode_score = 0
            obs = env.reset()
            while not done:
                obs_pretty = obs.reshape(1,  obs.shape[0])
                action_val = tentative_action.eval(feed_dict={X_placeholder: obs_pretty})
                new_obs, reward, done, info = env.step(action_val[0][0])
                if done and step < 499:
                    reward = -1
                else:
                    reward = 0
                new_obs_pretty = new_obs.reshape(1, new_obs.shape[0])

                estimated_value_obs = sess.run(value_function, feed_dict={X_placeholder: obs_pretty})
                estimated_value_obs = estimated_value_obs[0][0]
                estimated_value_new_obs = sess.run(value_function, feed_dict={X_placeholder: new_obs_pretty})
                estimated_value_new_obs = estimated_value_new_obs[0][0]
                target_estimated_value_obs = reward + discount_rate * estimated_value_new_obs
                advantage = target_estimated_value_obs - estimated_value_obs

                #######################
                # TODO: combiner les 2 appels

                #sess.run(training_op_actor, feed_dict={X_placeholder: obs_pretty)  # something with the advantage
                #

                loss_crit, _ = sess.run([loss_critic, training_op_critic], feed_dict={X_placeholder: obs_pretty,
                                                        target_value_placeholder: target_estimated_value_obs})
                print("episode: " + str(episode) + ", step: " + str(step) + ", loss_crit:" + str(loss_crit))
                ########################

                step += 1
                episode_score += reward
                obs = new_obs

            if episode % save_iterations == 0:  # periodical saves just in case
                saver.save(sess, tf_log_file)
        # Final save
        saver.save(sess, tf_log_file)

def run_episode_a2c_neural_network(environment,
                                   save_folder,
                                   show_renderer=False):
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(save_folder, "a2c_neural_net.ckpt.meta"))
        saver.restore(sess, tf.train.latest_checkpoint(save_folder))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        estimated_action_value = graph.get_tensor_by_name("actor/estimated_action_value:0")

        obs = environment.reset()
        done = False
        final_score = 0
        while not done:
            if show_renderer:
                environment.render()
            obs_pretty = obs.reshape(1, obs.shape[0])
            estimated_action = estimated_action_value.eval(feed_dict={X: obs_pretty})
            estimated_action = estimated_action[0][0]
            if estimated_action > 0.5:
                action = 1
            else:
                action = 0
            obs, reward, done, info = environment.step(action)
            final_score += reward
    environment.close()
    return final_score

def evaluate(environment, save_folder, nb_episodes=20, show_renderer=True):
    scores = []
    for i in range(0, nb_episodes):
        episode_score = run_episode_a2c_neural_network(environment, save_folder, show_renderer)
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
                               "..", "save", "a2c_neural_net")
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # Training
    train_policy_gradient_reinforced_neural_network(env, save_folder)

    # Evaluation of the imitation neural network
    # Hopefully it should perform as well as the human crafted policy
    mean_reward = evaluate(env, save_folder)
    print("Mean reward per episode: " + str(mean_reward))