# coding: utf-8

import matplotlib.pyplot as plt
import gym
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.WARN)  # Remove Info logs


def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()

    if reward_std != 0:
        return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]
    else:
        return [discounted_rewards for discounted_rewards in all_discounted_rewards]

def train_policy_gradient_reinforced_neural_network(environment,
                                                    save_folder,
                                                    nb_training_sessions=1000,
                                                    nb_episodes_per_training_session=10,
                                                    save_iterations=10,
                                                    discount_rate=0.95):
    tf.reset_default_graph()

    nb_neurons_input = environment.observation_space.shape[0]
    nb_neurons_hidden_layer = 4
    nb_neurons_output = 1
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, shape=[None, nb_neurons_input], name="X")

    with tf.name_scope("dnn"):
        initializer_w = tf.contrib.layers.xavier_initializer()
        initializer_b = tf.zeros_initializer()

        hidden = tf.layers.dense(X, nb_neurons_hidden_layer, \
                                 activation=tf.nn.elu, \
                                 kernel_initializer=initializer_w, \
                                 bias_initializer=initializer_b, \
                                 name="hidden")
        logits = tf.layers.dense(hidden, nb_neurons_output,
                                 name="logits")
        estimated_action_value = tf.nn.sigmoid(logits, name="estimated_action_value")  # 0 => left, 1 => right
        p_left_and_right = tf.concat(axis=1, values=[1 - estimated_action_value, estimated_action_value])
        tentative_action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

    y = tf.to_float(tentative_action)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(cross_entropy)
    gradients = [grad for grad, variable in grads_and_vars]
    gradient_placeholders = []
    grads_and_vars_feed = []
    for grad, variable in grads_and_vars:
        gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
        gradient_placeholders.append(gradient_placeholder)
        grads_and_vars_feed.append((gradient_placeholder, variable))
    training_op = optimizer.apply_gradients(grads_and_vars_feed)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    tf_log_file = os.path.join(save_folder, "policy_gradient_reinforced_neural_net.ckpt")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init.run()  # TODO: restore the previous network which imitates the human behaviour
        for training_session in tqdm(range(nb_training_sessions)):
            all_rewards = []
            all_gradients = []
            for episode in range(nb_episodes_per_training_session):
                current_rewards = []
                current_gradients = []
                obs = env.reset()
                done = False
                step = 0
                while not done:
                    action_val, gradients_val = sess.run([tentative_action, gradients],
                                                         feed_dict={X: obs.reshape(1, nb_neurons_input)})
                    obs, reward, done, info = env.step(action_val[0][0])
                    if done and step < 499:
                        reward = -1
                    else:
                        reward = 0
                    step += 1
                    current_rewards.append(reward)
                    current_gradients.append(gradients_val)
                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)
                print("End of episode: " + str(step))

            all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)

            feed_dict = {}
            for var_index, gradient_placeholder in enumerate(gradient_placeholders):
                mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                          for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
                feed_dict[gradient_placeholder] = mean_gradients
            sess.run(training_op, feed_dict=feed_dict)
            if training_session % save_iterations == 0:  # periodical saves just in case
                saver.save(sess, tf_log_file)
        # Final save
        saver.save(sess, tf_log_file)

def run_episode_policy_gradient_reinforced_neural_network(environment,
                                                          save_folder,
                                                          show_renderer=False):
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(save_folder, "policy_gradient_reinforced_neural_net.ckpt.meta"))
        saver.restore(sess, tf.train.latest_checkpoint(save_folder))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        estimated_action_value = graph.get_tensor_by_name("dnn/estimated_action_value:0")

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
        episode_score = run_episode_policy_gradient_reinforced_neural_network(environment, save_folder, False)
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
                               "..", "save", "policy_gradient_reinforced_neural_network")
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # Training
    train_policy_gradient_reinforced_neural_network(env, save_folder)

    # Evaluation of the imitation neural network
    # Hopefully it should perform as well as the human crafted policy
    mean_reward = evaluate(env, save_folder)
    print("Mean reward per episode: " + str(mean_reward))