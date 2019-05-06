# coding: utf-8

import matplotlib.pyplot as plt
import gym
import numpy as np
import os
import shutil
import tensorflow as tf
import tempfile
from random import randint
tf.logging.set_verbosity(tf.logging.WARN)  # Remove Info logs

NUMBER_OF_COMPETING_AGENTS = 20
SURVIVAL_REQUIREMENT = 5  # only the SURVIVAL_REQUIREMENT best will survive to the next stage
NUMBER_SELECTION_ROUNDS = 5

class Competitor:
    competitor_index = 0
    def __init__(self, environment):
        id = Competitor.competitor_index
        Competitor.competitor_index += 1
        tmp_folder = tempfile.gettempdir()
        competitor_folder = os.path.join(tmp_folder, "cartpole_arena", "competitor_" + str(id))
        if not os.path.isdir(competitor_folder):
            os.makedirs(competitor_folder)
        train_random_neural_network(environment, competitor_folder)
        self.id = id
        self.tmp_folder = competitor_folder
        self.env = environment
    def go_to_valhalla(self):
        print("goodbye " +str(self.id))
        if os.path.isdir(self.tmp_folder):
            if (self.tmp_folder).startswith(tempfile.gettempdir()):  # better safe than sorry
                shutil.rmtree(self.tmp_folder)
    def get_id(self):
        return self.id
    def get_tmp_folder(self):
        return self.tmp_folder
    def get_env(self):
        return self.env

def rank_competitors(competitors):
    scores = []
    for competitor in competitors:  # TODO: a paralleliser
        mean_reward = evaluate(competitor.get_env(), competitor.get_tmp_folder(), nb_episodes=50,
                               show_renderer=False, show_plot=False)
        scores.append(mean_reward)
        print("competitor " + str(competitor.get_id()) + " - " + str(mean_reward))

    sort_index = np.argsort(scores)  # The list is ordered from the worst to the best
    sort_index = np.flip(sort_index)
    ranked_competitors = competitors[sort_index]
    return ranked_competitors

def mutate_competitor(competitor_to_mutate, learning_rate=0.1):
    environment = competitor_to_mutate.get_env()
    new_competitor = Competitor(environment)

    # TODO: load weights of competitor_to_mutate and mutate them
    # have a learning rate that varies with the selection round
    # comme pour le recuit simulé ? D'abord beaucoup, puis moins, puis beaucoup, de manière cyclique ?
    ####################################
    # initialize random weights
    # curr_weights = []
    # curr_bias = []
    # # loop through each layer in the first network to get shapes
    # for l in range(1, len(networks[0].model.layers)):
    #     # get shapes of weight and bias layers
    #     bias_shape = np.array(networks[0].model.layers[l].get_weights()[1]).shape
    #     shape = np.array(networks[0].model.layers[l].get_weights()[0]).shape
    #
    #     # get the current weights of the first network as a baseline
    #     # init biases to 0 is we're not adjusting them
    #     N = networks[0].model.layers[l].get_weights()[0]
    #     if APPLY_BIAS:
    #         B = networks[0].model.layers[l].get_weights()[1]
    #     else:
    #         B = np.zeros(shape[1])
    #
    #     # add to containers
    #     curr_weights.append(N)
    # curr_bias.append(B)
    ########################################

    return new_competitor

def create_new_generation_of_competitors(ranked_competitors):
    for index in range(SURVIVAL_REQUIREMENT, NUMBER_OF_COMPETING_AGENTS):
        current_competitor = ranked_competitors[index]
        current_competitor.go_to_valhalla()
        random_index = randint(0, SURVIVAL_REQUIREMENT)
        competitor_id_to_mutate = ranked_competitors[random_index]
        new_competitor = mutate_competitor(competitor_id_to_mutate)
        ranked_competitors[index] = new_competitor
    return ranked_competitors

def darwinian_selection(environment):
    assert SURVIVAL_REQUIREMENT < NUMBER_OF_COMPETING_AGENTS

    # Create seed agents (complete random)
    competitors = []
    for i in range(0, NUMBER_OF_COMPETING_AGENTS):
        new_competitor = Competitor(environment)
        competitors.append(new_competitor)
        print("New competitor: " + str(new_competitor.get_id()))
    competitors = np.array(competitors)

    for round in range(0, NUMBER_SELECTION_ROUNDS):
        print("New round - " + str(round))
        ranked_competitors = rank_competitors(competitors)
        competitors = create_new_generation_of_competitors(ranked_competitors)

    champ = competitors[0]
    return champ

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

def evaluate(environment, save_folder, nb_episodes=20,
             show_renderer=True,
             show_plot=True):
    scores = []
    for i in range(0, nb_episodes):
        episode_score = run_episode_random_neural_network(environment, save_folder, show_renderer)
        scores.append(episode_score)

    mean_reward = np.mean(scores)
    if show_plot:
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
    champ = darwinian_selection(env)

    # save champ to the official save_folder
    for file in os.listdir(champ.get_tmp_folder()):
        shutil.copy(os.path.join(champ.get_tmp_folder(), file), save_folder)
        print("PLIP " + file)
        if file == "checkpoint":
            print("PLOP")
            with open(os.path.join(save_folder, file), 'w') as f:
                new_checkpoint_file = os.path.join(save_folder, "random_neural_net.ckpt")
                line_1 = 'model_checkpoint_path:  "{}"'.format(new_checkpoint_file)
                line_2 = 'all_model_checkpoint_paths:  "{}"'.format(new_checkpoint_file)
                print(line_1)
                f.write(line_1)
                f.write("\n")
                f.write(line_2)

    # Evaluation
    mean_reward = evaluate(env, save_folder)
    print("Mean reward per episode: " + str(mean_reward))


