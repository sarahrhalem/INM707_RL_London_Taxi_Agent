# -*- coding: utf-8 -*-
"""
@author: Sarah Rhalem
The implematation of a solver based on Q Learning algorithm

"""


import math
import numpy as np
import City
from BaseAgent import BaseAgent

class QLAgent(BaseAgent):
    def __init__(self,env,policy,alpha,gamma,max_episodes):
        self.a = alpha
        self.g = gamma
        self.q_table = np.zeros((env.n_states,4))
        self.env = env
        self.max_episodes = max_episodes
        self.policy  = policy
        self.error = np.zeros((env.n_states,4))+100
        self.total_rewards = []
        self.steps         = []

    def reset(self):
        self.q_table = np.zeros((self.env.n_states,4))
        #self.visited = np.zeros(self.env.n_states())
        self.error = np.zeros((self.env.n_states,4))+100
        self.policy.reset()
        #performance metrics
        self.total_rewards = []
        self.steps         = []
        
        return
    
    def train(self):
        self.reset()
        
        for episode in range(self.max_episodes):
            done = False
            current_state = self.env.reset()
            new_state = current_state
            episode_reward = 0
            episode_steps = 0
            while (not done):
                action_idx = self.policy(self.q_table[current_state['state_idx']])
            
                r,done,new_state = self.env.step(self.env.action_list[action_idx])
                q=self.q_table[current_state['state_idx'],action_idx]
                
                new_q = q + self.a*(r + self.g*np.max(self.q_table[new_state['state_idx']])-q)
                self.q_table[current_state['state_idx'],action_idx] = new_q
                self.error[current_state['state_idx'],action_idx]=abs(q-new_q)
                current_state = new_state
                
                episode_reward += r
                episode_steps +=1
            
            self.policy.update()
            
            self.total_rewards+=[episode_reward]
            self.steps +=[episode_steps/self.env.max_steps]
            
        return
    
    def next_action(self,state):
        action_idx = np.argmax(self.q_table[state['state_idx']])
        return self.env.action_list[action_idx]
    
    def get_output(self,output='reward',is_avg=True,sample=1):
        data = []
        if(output=='steps'):
            data = self.steps
        else:
            data = self.total_rewards
            
        return [sum(data[:idx+1][-sample:])/len(data[:idx+1][-sample:]) for idx in range(len(data))]
    
class EGPolicy:
    def __init__(self,epslion,decay,seed):
        self.epslion = epslion
        self.current_epsilon = epslion
        self.decay = decay
        self.rng   = np.random.default_rng(seed=seed)
        return
    
    def update(self):
        self.current_epsilon *= self.decay
        return
    def reset(self):
        self.current_epsilon = self.epslion
        return
    
    def __call__(self,qv):
        is_random = self.rng.random()<self.current_epsilon
        if np.max(qv)==np.min(qv):#force random choice when there is no clear argmax
            is_random=True
        if is_random:
            return self.rng.choice(range(4))
        else:
            return np.argmax(qv)
    
    

