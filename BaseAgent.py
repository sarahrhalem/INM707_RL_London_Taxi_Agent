# -*- coding: utf-8 -*-
"""
@author: Sarah Rhalem

A class for base Agent that can act following simple policies
The set of base policies will be roads (with 'in or 'out' actions),zone (with 'cw' or 'acw') and random
policy that combine the two base ones and a random move
"""
import Visualisation
import City
import numpy as np
import math

class BaseAgent:
    def __init__(self,env,policy):
        self.policy = policy
        self.env    = env
    def evaluate(self,episodes,sample=1):
        total_rewards=[]
        steps=[]
        
        for episode in range(episodes):

            state_dict=self.env.reset()
            done=False
            total_reward = 0
            n_steps = 0

            while(not done):
                a = self.next_action(state_dict)
                
                reward,done,new_state_dict = self.env.step(a)     
                total_reward+=reward
                n_steps+=1
                state_dict=new_state_dict

            total_rewards+=[total_reward]

            steps+=[n_steps/self.env.max_steps]

        avg_rewards = [sum(total_rewards[:idx+1][-sample:])/len(total_rewards[:idx+1][-sample:]) for idx in range(len(total_rewards))]
        avg_steps   = [sum(steps[:idx+1][-sample:])/len(steps[:idx+1][-sample:]) for idx in range(len(steps))]
        
        return avg_rewards,avg_steps
    
    def next_action(self,state):
        return self.policy(state,self.env.action_list)
    
    def single_exp(self,vis=False,show_traffic=False,fig_size=10):
        
        state_dict=self.env.reset()
        vis = self.env.get_vis()
        
        vis['trajectory_road']=[]
        vis['trajectory_zone']=[]
        prev_loc = self.env.current_loc
        prev_loc_c = self.env.get_current_loc_cooridnates()
        done=False
        total_reward = 0
        n_steps = 0
        actions= []
        traffic_jam = [state_dict['taffic']]
        while(not done):
            
            a = self.next_action(state_dict)
                
            reward,done,new_state_dict = self.env.step(a) 
            
            if a in ['in','out']:
                vis['trajectory_road']+=[[self.env.get_current_loc_cooridnates(),prev_loc_c]]
            elif a=='acw':
                vis['trajectory_zone']+=[[sum(self.env.l_radius[:self.env.current_loc.radius_idx+1])*2,
                         360*prev_loc.angle_idx/self.env.n_stations,360/self.env.n_stations]]
            else:
                vis['trajectory_zone']+=[[sum(self.env.l_radius[:self.env.current_loc.radius_idx+1])*2,
                         360*self.env.current_loc.angle_idx/self.env.n_stations,360/self.env.n_stations]]
            
            prev_loc = self.env.current_loc
            prev_loc_c = self.env.get_current_loc_cooridnates()
            actions+=[a]
            total_reward+=reward
            n_steps+=1
            state_dict=new_state_dict
            traffic_jam += [state_dict['taffic']]
 
        if vis:
            Visualisation.plot_vis(vis,fig_size)
        if(show_traffic):
            return total_reward,n_steps,actions,traffic_jam
        else:
            return total_reward,n_steps,actions
    
class RoadPolicy:
     
    def __call__(self,state,action_list):

        current_loc = state['current_loc']
        target_loc = state['target']
        
        
        next_loc = current_loc.next_station('in')
        
        current_dist = math.cos(2*math.pi*(current_loc.angle_idx-target_loc.angle_idx)/target_loc.n_stations)
        next_dist    = math.cos(2*math.pi*(next_loc.angle_idx-target_loc.angle_idx)/target_loc.n_stations)
        
        if current_dist>0:
            current_dist*=abs(current_loc.radius_idx-target_loc.radius_idx+1)
        else:
            current_dist*=current_loc.radius_idx+target_loc.radius_idx
        
        if next_dist>0:
            next_dist*=abs(next_loc.radius_idx-target_loc.radius_idx+1)
        else:
            next_dist*=next_loc.radius_idx+target_loc.radius_idx

        if (abs(next_dist)<abs(current_dist)):
            return 'in'
        else:
            return 'out'

class ZonePolicy:
       
    def __call__(self,state,action_list):
        current_loc = state['current_loc']
        target_loc = state['target']

        current_angle = current_loc.angle_idx
        next_angle = current_loc.next_station('cw').angle_idx
        
        angle_tgt = min((current_angle-target_loc.angle_idx)%target_loc.n_stations,
                        (target_loc.angle_idx-current_angle)%target_loc.n_stations)
        
        nex_angle_tgt = min((next_angle-target_loc.angle_idx)%target_loc.n_stations,
                        (target_loc.angle_idx-next_angle)%target_loc.n_stations)
        if(nex_angle_tgt>angle_tgt):
            return 'acw'
        else:
            return 'cw'
    
class RandomPolicy:
    def __init__(self,p=[0.45,0.45,0.1],seed=0):
        #p is the probability to return RoadPolicy, the probability to return zone or random
        #is (1-p)/2
        self.rng   = np.random.default_rng(seed=seed)
        self.p = p
    def __call__(self,state,action_list):
        x = self.rng.choice(['road','zone','random'],p=self.p)
        if x=='road':
            policy = RoadPolicy()
            return policy(state,action_list)
        elif x=='zone':
            policy = ZonePolicy()
            return policy(state,action_list)
        else:
            return self.rng.choice(action_list)