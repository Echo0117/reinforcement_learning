"""
This is the machinnery that runs your agent in an environment.

This is not intented to be modified during the practical.
"""

class Runner:
    def __init__(self, environment, agent, verbose=False):
        self.environment = environment
        self.agent = agent
        self.verbose = verbose

    def step(self):
        action = self.agent.choose()
        reward = self.environment.act(action)
        self.agent.update(action, reward)
        return (action, reward)

    def loop(self, iterations):
        cumul_reward = 0.0
        list_cumul = []
        for i in range(1, iterations+1):
            (act, rew) = self.step()
            cumul_reward += rew
            if self.verbose:
                print("Simulation step {}:".format(i))
                print(" ->            action: {}".format(act))
                print(" ->            reward: {}".format(rew))
                print(" -> cumulative reward: {}".format(cumul_reward))
            list_cumul.append(cumul_reward)
        return cumul_reward, list_cumul

def iter_or_loopcall(o, count,n_actions):
    if callable(o):
        return [ o(n_actions) for _ in range(count) ]
    else:
        # must be iterable
        return list(iter(o))

class BatchRunner:
    """
    Runs several instances of the same RL problem in parallel
    and aggregates the results.
    """

    def __init__(self, env_maker, agent_maker, count,n_actions, verbose=False):
        self.environments = iter_or_loopcall(env_maker, count,n_actions)
        self.agents = iter_or_loopcall(agent_maker, count,n_actions)
        assert(len(self.agents) == len(self.environments))
        self.verbose = verbose
        

    def step(self):
        actions = [ agent.choose() for (agent) in self.agents ]
        rewards = [ env.act(action) for (env, action) in zip(self.environments, actions) ]
        for (agent, action, reward) in zip(self.agents, actions, rewards):
            agent.update(action, reward)
        return sum(rewards)/len(rewards)

    def loop(self, iterations):
        cum_avg_reward = 0.0
        list_cumul = []
        for i in range(1, iterations+1):
            avg_reward = self.step()
            cum_avg_reward += avg_reward
            if self.verbose:
                print("Simulation step {}:".format(i))
                print(" ->            average reward: {}".format(avg_reward))
                print(" -> cumulative average reward: {}".format(cum_avg_reward))
            list_cumul.append(cum_avg_reward)
        return cum_avg_reward, list_cumul
