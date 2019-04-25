# coding: utf-8

import matplotlib.pyplot as plt
import gym
import numpy as np
import bisect
import math
import os
import random


def obs_2_row_index(obs,
                    granularity,
                    position_min=-2.4, position_max=2.4,
                    velocity_min=-3, velocity_max=3,
                    angle_min=-0.3, angle_max=0.3,
                    angular_velocity_min=-2, angular_velocity_max=2
                    ):
    assert granularity >= 2
    assert isinstance(granularity, int)

    position, velocity, angle, angular_velocity = obs

    discretized_positions = np.linspace(position_min, position_max, num=granularity - 1)
    discretized_velocities = np.linspace(velocity_min, velocity_max, num=granularity - 1)
    discretized_angles = np.linspace(angle_min, angle_max, num=granularity - 1)
    discretized_angular_velocities = np.linspace(angular_velocity_min, angular_velocity_max, num=granularity - 1)

    position_index = bisect.bisect_left(discretized_positions, position)
    velocity_index = bisect.bisect_left(discretized_velocities, velocity)
    angle_index = bisect.bisect_left(discretized_angles, angle)
    angular_velocity_index = bisect.bisect_left(discretized_angular_velocities, angular_velocity)

    row_index = math.pow(granularity, 0) * position_index \
                + math.pow(granularity, 1) * velocity_index \
                + math.pow(granularity, 2) * angle_index \
                + math.pow(granularity, 3) * angular_velocity_index
    row_index = int(row_index)

    return row_index

def train_q_table(environment,
                  save_folder,
                  granularity=16,
                  nb_training_episodes=5000,
                  learning_rate=0.4,
                  discount_rate=1,  # deterministic environment
                  epsilon_exploration=0.5,
                  exploration_decay_rate=0.999,
                  min_exploration_rate=0.01,
                  target_nb_moves=500):

    nb_columns = environment.action_space.n
    observation_space_dimension = environment.observation_space.shape[0]  # Here the obs is a vector of 4 dimension
    nb_rows = pow(granularity, observation_space_dimension)

    q_table = np.zeros([nb_rows, nb_columns])  # our table of Q values
    print("The Q-table is of shape:" + str(q_table.shape))
    nb_times_visited = np.zeros([nb_rows, nb_columns])

    episode_scores_record = []  # Serves to follow how well the agent learns
    for episode in range(0, nb_training_episodes):
        step = 0
        done = False
        episode_score, reward = 0, 0
        obs = env.reset()
        while not done:
            row_index = obs_2_row_index(obs, granularity)
            if random.uniform(0, 1) > epsilon_exploration:
                action = np.argmax(q_table[row_index])
            else:
                action = np.random.randint(0, 2)
            col_index = action
            new_obs, reward, done, info = env.step(action)
            episode_score += reward
            if done:
                reward = step - target_nb_moves
            new_row_index = obs_2_row_index(new_obs, granularity)
            q_table[row_index, col_index] += learning_rate \
                                       * (reward + discount_rate * np.max(q_table[new_row_index]) - q_table[row_index, col_index])
            nb_times_visited[row_index, col_index] += 1

            obs = new_obs
            step += 1

        episode_scores_record.append(step)
        if episode % 50 == 0:
            print('Episode {} - score: {}'.format(episode, step))
        epsilon_exploration = max(min_exploration_rate, epsilon_exploration * exploration_decay_rate)  # TODO: min explo rate

    # training summary
    plt.plot(episode_scores_record)
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.show()

    # save table
    q_table_save_file = os.path.join(save_folder, "q_table_{}.npy".format(granularity))
    np.save(q_table_save_file, q_table)
    q_table_csv_save_file = os.path.join(save_folder, "q_table_{}.csv".format(granularity))
    np.savetxt(q_table_csv_save_file, q_table, delimiter=";")
    nb_times_visited_csv_save_file = os.path.join(save_folder, "nb_times_visited_{}.csv".format(granularity))
    np.savetxt(nb_times_visited_csv_save_file, nb_times_visited, delimiter=";")

def run_episode_q_learning(environment, q_table, granularity=16, show_renderer=False):
    obs = environment.reset()
    done = False
    final_score = 0
    while not done:
        if show_renderer:
            environment.render()
        row_index = obs_2_row_index(obs, granularity)
        action = np.argmax(q_table[row_index])
        obs, reward, done, info = environment.step(action)
        final_score += reward
    environment.reset()
    environment.close()
    return final_score

def load_q_table(q_table_npy_file):
    try:
        q_table = np.load(q_table_npy_file)
        return q_table
    except FileNotFoundError:
        print("file *{}* not found!!".format(q_table_npy_file))

def evaluate(environment, q_table, granularity, nb_episodes=20):
    scores = []
    for i in range(0, nb_episodes):
        episode_score = run_episode_q_learning(environment, q_table, granularity, True)
        scores.append(episode_score)

    mean_reward = np.mean(scores)
    plt.plot(scores)
    plt.title("Mean reward: {}".format(mean_reward))
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.show()


if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    granularity = 16  # will condition how we discretize the continuous variables
    save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "save")
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    q_table_npy_file = os.path.join(save_folder, "q_table_{}.npy".format(granularity))

    # Training
    train_q_table(env,
                  save_folder,
                  granularity=16,
                  nb_training_episodes=5000,
                  learning_rate=0.2,
                  discount_rate=1,
                  epsilon_exploration=0.5,
                  exploration_decay_rate=0.999,
                  min_exploration_rate=0.01,
                  target_nb_moves=500)

    # Loading the saved q_table
    q_table = load_q_table(q_table_npy_file)

    # Evaluation of the q_learning policy
    evaluate(env, q_table, granularity)