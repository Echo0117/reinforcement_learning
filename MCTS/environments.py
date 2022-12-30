import gym

class Environment_go:
    # import gym_go # TO UNCOMMENT  you will need to install this to launch a GO env ! 
   #https://github.com/aigagror/GymGo
    def __init__(self,size):
        """Instanciate a new environement in its initial state.
        """
        self.env =  gym.make('gym_go:go-v0', size=size, komi=0, reward_method='real')
        self.s = self.env.reset()
        self.n_a = self.env.action_space.n
    
    def act(self, action):
        """Perform given action by the agent on the environment,
        and returns a reward.
        """
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info 
    def turn(self):
        "Give the current player turn"
        return self.env.turn()
    
    def invalid_moves(self):
        "Return non-feasable moves"
        return self.env.info()['invalid_moves']
    
    def uniform_random_action(self):
        "Return a random move from uniform distribution on feasable moves"
        return self.env.uniform_random_action()
    def is_invalid(self,action):
        "Return a boolean indication if the selected action is non-feasable"
        return self.invalid_moves()[action]
    def reset(self):
        self.env.reset()
    def render(self):
        self.env.render('terminal')

    def get_current_state(self):
        return self.s
    def is_done(self):# is the game finished ?
        return self.env.game_ended()

    def display(self): # return the string of the ascii rendering of the board
        return self.env.render('terminal')    


from gym import spaces
import numpy as np
import random


class Environment_tic_tac_toe:
    def __init__(self):
        """Instanciate a new environement in its initial state.
        """
        self.env =  gym_tic_tac_toe()
        self.s = self.env.reset()
        self.n_a = self.env.action_space.n
        self.init_turn = int(self.env.turn)
      
    def act(self, action):
        """Perform given action by the agent on the environment,
        and returns a reward.
        """
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info 
    def get_current_state(self):
        return self.env.observe()   
    def turn(self): #Give current turn
        return self.env.get_info()['turn']
    def is_done(self):# is the game finished ?
        return self.env.done
    def invalid_moves(self): # return the array of corresponding invalid moves
        return self.env.get_info()['invalid_moves']
    def uniform_random_action(self): # Sample uniformly a random action from feasable actions
        return self.env.uniform_random_action()
    def is_invalid(self,action): # is the current action feasable ?
        return self.invalid_moves()[action]
    def reset(self):
        return self.env.reset()
    def render(self):
        print(self.env.render('terminal'))
    def display(self): # return the string of the ascii rendering of the board
        return self.env.render('terminal')

class gym_tic_tac_toe(gym.Env):
    metadata = {'render_modes' :['terminal']}

    def __init__(self):
        self.action_space = spaces.Discrete(9)
        self.reset()
        self.turn = np.random.randint(0,2)
        # First player O : 0, second player X :1
        self.n_states = 262144
    def uniform_random_action(self):
        index = self.action_space.sample()
        while self._get_invalid()[index]:
            index = self.action_space.sample()
        return index
    def reset(self):
        self.board = np.zeros(9*2).reshape(2,-1)
        self.turn = np.random.randint(0,2)
        self.done = False
        return self._get_obs()

    def place(self,action):
        if self._get_invalid()[action]:
            raise Exception("Not feasable action")
        else:
            self.board[self.turn,action] = 1
            # self.board[2,:] = (self.turn+1)%2
        
    def check_game_status(self):
        if check_line(self.board[0,:].reshape(3,3)):
            return 1
        if check_line(self.board[1,:].reshape(3,3)):
            return 2
        if np.all(self._get_invalid()):
            return 0
        else:
            return -1

    def step(self, action):
        assert self.action_space.contains(action)

        if self.done:
            return self._get_obs(), 0,True,None
        
        self.place(action)
        self.turn = (self.turn+1)%2
        status = self.check_game_status()
        reward=0
        if status>=0:
            self.done = True
            if status ==1: # First player win
                reward = 1
            if status ==2: # Second player win
                reward = -1
            
        return self._get_obs(),reward,self.done,self.get_info()
    
    def render(self,render_mode="None"):
        if render_mode == "terminal":
            string_board = self._show_board()  
            string_board+='\n'
            if self.done:
                string_board+="End of the game \n"
            else:
                string_board+='Turn : '+('O' if self.turn == 0 else 'X')+ "\n"
            return string_board
     
    def _show_board(self):
        """Draw tictactoe board."""
        board = np.full((3,3),' ')
        for i in range(3):
            for j in range(3):
                if self.board[0,i+j*3] == 1:
                    board[i,j] = 'O'
                if self.board[1,i+j*3] == 1:
                    board[i,j] = 'X'
        string_board = ' | '.join(board[0,:]) + "\n"
        string_board += ' | '.join(board[1,:]) + "\n"
        string_board +=' | '.join(board[2,:]) + "\n"
        return string_board

    def _get_obs(self):
        return (self.board,self.turn)
    def observe(self):
        return self._get_obs()  
    def _get_invalid(self):
        return np.logical_or(self.board[0,:],self.board[1,:])
    def get_info(self):
        return {'turn' : self.turn,'invalid_moves': self._get_invalid()}

    
def check_line(board):
    if np.all(board):
        return True
    for i in range(3):
        if np.all(board[i,:]) or np.all(board[:,i]):
            return True
    if np.all(np.multiply(board,np.eye(3)) == np.eye(3)):
        return True
    if np.all(np.multiply(board,np.rot90(np.eye(3))) == np.rot90(np.eye(3))):
        return True
    else:
        return False
