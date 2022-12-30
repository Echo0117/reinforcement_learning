"""
This is the machinnery that runs your agent in an environment.

This is not intented to be modified during the practical.
"""
import copy, gym
import matplotlib
import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_cost_to_go_mountain_car(env, type_value, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)


    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    if type_value == "state_value":
        Z = np.apply_along_axis(lambda _: estimator(_), 2, np.dstack([X, Y]))
        ax.set_title("State-value function: V(s)")
    else:
        Z = np.apply_along_axis(lambda _: np.max([estimator(_, a) for a in range(0, 1, 2)]), 2, np.dstack([X, Y]))
        ax.set_title("State-action value Function: $max_{a}q(s, a)$")

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1) # , cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    fig.colorbar(surf)
    plt.show()


class FARunner:
    def __init__(self, environment, agent, type_value, verbose=False):
        self.type_value = type_value
        self.environment = environment
        self.agent = agent
        self.verbose = verbose

    def step(self):
        observation = self.environment.observe()
        action = self.agent_to_act.act(observation)
        (reward, stop) = self.environment.default_act(action)
        new_observation = self.environment.observe()
        self.agent.update(observation, action, reward, new_observation, stop)
        return (observation, action, reward, stop, new_observation)

    def loop(self, games,max_iter):
        cumul_reward = 0.0
        list_n_episodes = []
        for g in range(1, games+1):
            print("Episode ", g)
            self.environment.reset()
            stop = False
            i = 0
            reward = 0
            self.agent_to_act = copy.copy(self.agent)
            while (not stop) and (i < max_iter):
                if self.verbose:
                    print("Simulation step {}:".format(i))
                (obs, act, rew, stop, new_observation) = self.step()
                cumul_reward += rew
                reward += rew
                if self.verbose:
                    print(" ->       game: {}".format(g))
                    print(" ->       observation: {}".format(obs))
                    print(" ->            action: {}".format(act))
                    print(" ->            reward: {}".format(rew))
                    print(" -> cumulative reward: {}".format(reward))
                    if stop:
                        print(" ->    Terminal event: {}".format(new_observation))
                        input()
                    print("")
                i += 1
            list_n_episodes.append(reward)

            if self.verbose:
                print(" <=> Finished episode number: {} <=>".format(g))
                print("")
        estimator = self.agent.v if self.type_value == "state_value" else self.agent.q
        plot_cost_to_go_mountain_car(self.environment.env, self.type_value, estimator)

        env = gym.make("MountainCar-v0")

        for _ in range(10):
            done = False
            observation = env.reset()
            while not done:
                env.render()
                observation, reward, done, info = env.step(self.agent.act(observation)) # take a random action
        env.close()

        return cumul_reward, list_n_episodes
