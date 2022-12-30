import argparse
import agents
import environment
import runner
import random
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--agent', metavar='AGENT_CLASS', choices=['RandomAgent', 'epsGreedyAgent', 'BesaAgent', 'SoftmaxAgent', 'UCBAgent', 'ThompsonAgent', 'KLUCBAgent'], default='RandomAgent', type=str, help='Class to use for the agent. Must be in the \'agent\' module. Possible choice: (RandomAgent, epsGreedyAgent, BesaAgent, SoftmaxAgent, UCBAgent, ThompsonAgent, KLUCBAgent)')
parser.add_argument('--niter', type=int, metavar='n', default='1000', help='number of iterations to simulate')
parser.add_argument('--batch', type=int, metavar='nagent', default=None, help='batch run several agent at the same time')
parser.add_argument('--verbose', action='store_true', help='Display cumulative results at each step')

random.seed(0)

def main():
    args = parser.parse_args()
    agent_class = eval('agents.{}'.format(args.agent))
    env_class = eval('environment.Environment')

    if args.batch is not None:
        print("Running a batched simulation with {} agents in parallel...".format(args.batch))
        my_runner = runner.BatchRunner(env_class, agent_class, args.batch, args.verbose)
        final_reward, list_cumul = my_runner.loop(args.niter)
        print("Obtained a final average reward of {}".format(final_reward))
    else:
        print("Running a single instance simulation...")
        my_runner = runner.Runner(env_class(), agent_class(), args.verbose)
        final_reward, list_cumul = my_runner.loop(args.niter)
        print("Obtained a final reward of {}".format(final_reward))

    plt.plot(list_cumul)
    plt.xlabel("Iter")
    plt.ylabel("Cum. Reward")
    plt.title("Agent: {}".format(args.agent))
    plt.show()

if __name__ == "__main__":
    main()
