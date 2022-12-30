import numpy as np

def obs_to_int(obs): # Convert our state representation  into an integer
    array,_=obs
    
    b = array.flatten()
    return int((b.dot(2**np.arange(b.size)[::-1])))   

class Runner: 
    def __init__(self, environment, agent1,agent2, verbose=False,training=True):
        self.environment = environment
        self.agent_1 = agent1
        self.agent_2 = agent2
        self.agent_1.training = training
        self.agent_2.training = False
        self.verbose = verbose
        self.a1 = 0
        self.s1 = 0
        self.s2 = 0
        self.display = None
    def step(self,turn):
        if turn == 0:
            action = self.agent_1.choose()
            self.a1 = action
            self.s1 = obs_to_int(self.environment.get_current_state())
            obs, reward, done, info = self.environment.act(action)
            if done:
                self.reward_1 = reward
                self.agent_1.update_terminal(self.s1,self.a1,self.reward_1)
            
            self.agent_2.observe(obs)
            return (obs, action, reward, done,info)
        else:
            action = self.agent_2.choose()
            observation, reward, done, info = self.environment.act(action)
            self.reward_1 = reward
            self.s2 = obs_to_int(observation)
            self.agent_1.update(self.s1,self.a1,self.s2,self.reward_1)
            self.agent_1.observe(observation)
            return (observation, action, reward, done,info)

    def loop(self, n_episodes):
        list_episode_reward = []
        self.agent_1.observe(self.environment.get_current_state())
        self.agent_2.observe(self.environment.get_current_state())
        if self.verbose:
            self.display=''
        for i in range(1, n_episodes + 1):
            if self.verbose:
                self.display+="\n ==== Game "+str(i)+ " ====\n"
            episode_reward = 0.0 
            done = False
            while not done:
                (obs, act, rew, done,info) = self.step(self.environment.turn())
                if self.verbose:
                    self.environment.env.render('terminal')
                    self.display+=self.environment.display()
            episode_reward = rew
            # Reset the env after an episode
            self.environment.reset()
            self.agent_1.observe(obs)
            self.agent_2.observe(obs)
            list_episode_reward.append(episode_reward)
        return list_episode_reward,self.display

