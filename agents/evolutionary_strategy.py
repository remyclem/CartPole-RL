# coding: utf-8

import matplotlib.pyplot as plt
import gym
import numpy as np
import os
import tensorflow as tf
import tempfile

tf.logging.set_verbosity(tf.logging.WARN)  # Remove Info logs

NUMBER_OF_COMPETING_AGENTS = 10
SURVIVAL_REQUIREMENT = 3  # only the 3 best will survive to the next stage
NUMBER_SELECTION_ROUNDS = 3

class Competitor:
    def __init__(self, id, environment):
        tmp_folder = tempfile.gettempdir()
        competitor_folder = os.path.join(tmp_folder, "cartpole_arena", "competitor_" + str(id))
        if not os.path.isdir(competitor_folder):
            os.makedirs(competitor_folder)
        train_random_neural_network(environment, competitor_folder)
        self.id = id
        self.tmp_folder = competitor_folder
        self.env = environment
    def go_to_valhalla(self):
        if os.path.isdir(self.tmp_folder):
            if (self.tmp_folder).startswith(tempfile.gettempdir()):  # better safe than sorry
                os.rmdir(self.tmp_folder)

def rank_competitors(competitors):

    for competitor in competitors:
        mean_reward = evaluate(competitor.env, competitor.tmp_folder, nb_episodes=20, show_renderer=False)
        print("competitor " + str(competitor) + " - " + str(mean_reward))

    survivor_ids = None
    return survivor_ids

def mutate_competitor(competitor_id):
    pass

def create_new_generation_of_competitors(survivor_ids):
    # the SURVIVAL_REQUIREMENT competitors in survivor_ids are not changed
    # 2 * SURVIVAL_REQUIREMENT are mutations of those in survivor_ids
    # the rest is created at random
    pass

def darwinian_selection(environment):
    assert SURVIVAL_REQUIREMENT < NUMBER_OF_COMPETING_AGENTS

    # Create seed agents (complete random)
    competitors = []
    for competitor_id in range(0, NUMBER_OF_COMPETING_AGENTS):
        new_competitor = Competitor(competitor_id, environment)
        competitors.append(new_competitor)

    for round in range(0, NUMBER_SELECTION_ROUNDS):
        survivor_ids = rank_competitors(competitors)
        create_new_generation_of_competitors(survivor_ids)

    champ_folder = None
    return champ_folder

def train_random_neural_network(environment, save_folder):
    tf.reset_default_graph()

    nb_neurons_input = environment.observation_space.shape[0]
    nb_neurons_hidden_layer = 4
    nb_neurons_output = 1

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

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init.run()
        tf_log_file = os.path.join(save_folder, "random_neural_net.ckpt")
        saver.save(sess, tf_log_file)

def run_episode_random_neural_network(environment,
                                      save_folder,
                                      show_renderer=False):
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(save_folder, "random_neural_net.ckpt.meta"))
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
        episode_score = run_episode_random_neural_network(environment, save_folder, show_renderer)
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
                               "..", "save", "evolutionary_algo")
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # Training
    champ_folder = darwinian_selection(env)

    # save champ to the official save_folder



    # Evaluation of the imitation neural network
    # Hopefully it should perform as well as the human crafted policy
    #mean_reward = evaluate(env, save_folder)
    #print("Mean reward per episode: " + str(mean_reward))
