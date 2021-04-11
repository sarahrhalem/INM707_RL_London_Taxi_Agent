# -*- coding: utf-8 -*-
"""
@author: Sarah Rhalem

The implementation of soft actor critic solver in discrete case and with a Experience replay memory  
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import time
import collections
import numpy as np

import matplotlib.pyplot as plt

import City

# a collection for the iterms needed in the memory

Transition = collections.namedtuple('Transition',field_names=['state_goal','action','reward','done','new_state_goal'])
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Transforming environement input/output to pytorch tensor
def value_to_tensor(value,device=device):
    return torch.tensor([float(value)], device=device).unsqueeze(0)

def state_goal_to_tensor(state_dict,device):
    return torch.tensor(state_dict['observation'], device=device).unsqueeze(0)

#Replay buffer with fixed capacity
class ReplayBuffer():
    def __init__ (self, capacity,device=device,seed=0):
        self.device = device
        self.rng = np.random.default_rng(seed=seed)
        self.memory = collections.deque(maxlen=capacity)
    def __len__(self):
        return len(self.memory)
    
    def push(self,state_dict,action,reward,done,new_state_dict):
        transition = Transition(state_goal_to_tensor(state_dict,self.device),
                                value_to_tensor(action,self.device),
                                value_to_tensor(reward,self.device),
                                value_to_tensor(done,self.device),
                                state_goal_to_tensor(new_state_dict,self.device))
        self.memory.append(transition)
        return
    def is_full(self):
        return (len(self.memory)==self.memory.maxlen)
    def sample(self,batch_size):
        indices = self.rng.choice(len(self.memory),batch_size,replace=False)
        states_goals,actions,rewards,dones,new_states_goals =zip(*[self.memory[idx] for idx in indices])
        return torch.cat(states_goals),torch.cat(actions),torch.cat(rewards),torch.cat(dones),torch.cat(new_states_goals)
    


# a Policy Neural Netowrk to represent policy parameters with two hidden layers, double ReLu and soft max output
class PolicyNN(nn.Module):  
    def __init__(self, n_inputs, size_hidden, n_actions):
        
        super(PolicyNN,self).__init__()
        self.fc1= nn.Linear(n_inputs, size_hidden[0])
        self.fc2= nn.Linear(size_hidden[0], size_hidden[1])
        self.fc3= nn.Linear(size_hidden[1], n_actions)
        
        #ensure uniform distrubition at initialization
        nn.init.constant_(self.fc1.weight,1e-4)
        nn.init.constant_(self.fc2.weight,1e-4)
        nn.init.constant_(self.fc3.weight,1e-4)
        
        nn.init.constant_(self.fc1.bias,1)
        nn.init.constant_(self.fc2.bias,1)
        nn.init.constant_(self.fc3.bias,1)
        

    def forward(self, x):
        q1= F.relu(self.fc1(x.float()))
        q1= F.relu(self.fc2(q1))
        q1= self.fc3(q1)
        
        output = F.softmax(q1, dim=1)
        return output  

# Q Neural Network to represent Q values two hidden layers and double ReLu
class SingleQNN(nn.Module):  
    def __init__ (self, n_inputs, size_hidden , n_actions):
        super(SingleQNN, self).__init__()

        self.fc1= nn.Linear(n_inputs, size_hidden[0])
        self.fc2= nn.Linear(size_hidden[0], size_hidden[1])
        self.fc3= nn.Linear(size_hidden[1], n_actions)

    def forward(self, x):

        q1= F.relu(self.fc1(x.float()))
        q1= F.relu(self.fc2(q1))
        q1= self.fc3(q1)
        
        return q1

class DoubleQNN(nn.Module):
    def __init__ (self, n_inputs, size_hidden , n_actions):
        super(DoubleQNN, self).__init__()
        
        self.qnn1 = SingleQNN(n_inputs, size_hidden , n_actions)
        self.qnn2 = SingleQNN(n_inputs, size_hidden , n_actions)
        
    def forward(self, x):
        q1 = self.qnn1.forward(x)
        q2 = self.qnn2.forward(x)
        return q1,q2

# a function to select an item from a list based on probabilities input and a noise with normal distribution
   
def random_choice(value_list,p,rng,epsilon):
    x = rng.random()+epsilon*rng.normal()
    
    if x>1:
        return 0,value_list[0]
    elif x>sum(p[:3]):
        return 3,value_list[3]
    elif  x>sum(p[:2]):
        return 2,value_list[2]
    elif x>p[1]:
        return 1,value_list[1]
    elif x>0:
        return 0,value_list[0]
    else:
        return 3,value_list[3]
    
#The implementation of Soft Actor critic agent that performs three steps:
    # fill initial memory with randomly generated values
    # train the agent by adding optimizing q value and the policies entropy
    # generate a test of the environment based on final values
    
class SACD_Agent:

    def __init__(self,env,capacity=100000,batch_size=100,n_episodes=1000,epsilon=0.01,epsilon_decay=1,
                 q_param_size_hidden=[256,256],policy_param_size_hidden=[256,256],alpha=0.1,
                 q_lr=1e-3,policy_lr=1e-3,gamma=1,update_interval = 4,device=device, seed=0):
        
        torch.manual_seed(seed)
        
        self.env             = env
        self.action_list     = env.action_list
        self.n_episodes      = n_episodes # max number of episodes for training
        self.batch_size      = batch_size
        self.gamma           = gamma
        self.device          = device
        
        self.epsilon         = epsilon
        self.current_epsilon = epsilon     
        self.epsilon_decay   = epsilon_decay
        
        self.update_interval = update_interval
        
        # networks for q values
        self.replay_buffer = ReplayBuffer(capacity,device,seed)
        
        self.rng   = np.random.default_rng(seed=seed)
        
        self.alpha = alpha #entropy coefficient
        
        self.qn            = DoubleQNN(self.env.state_dim,q_param_size_hidden,len(self.action_list))
        self.qn1_optimizer = torch.optim.Adam(self.qn.qnn1.parameters(),lr=q_lr)
        self.qn2_optimizer = torch.optim.Adam(self.qn.qnn2.parameters(),lr=q_lr)
        self.qn.to(device)
        self.qtarget       = DoubleQNN(self.env.state_dim,q_param_size_hidden,len(self.action_list))
        self.qtarget.to(device)
        self.qtarget.load_state_dict(self.qn.state_dict())
        self.qtarget.eval()
        
        #network for policy
        self.policy       = PolicyNN(self.env.state_dim,policy_param_size_hidden,len(self.action_list))
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(),lr=policy_lr)
        self.policy.to(device)
        #outputs and performance checks
        self.total_rewards = []
        self.steps         = []
        self.qlosses       = []
        self.policy_losses = []
    
    def get_action(self,state):#get a random action at training phase base on the policy and normal noise with epsilon as standard deviation
        self.policy.eval()
        p = self.policy(state).cpu().detach().numpy()[0]
        a_idx,a = random_choice(self.action_list,p,self.rng,self.current_epsilon)
        self.current_epsilon *= self.epsilon_decay
        self.policy.train()
        return a_idx,a

    def fill_memory(self):
    
        while(not self.replay_buffer.is_full()):
            
            state_dict=self.env.reset()
            done=False
            total_reward = 0
            n_steps = 0
             
            while(not done):
                state_tensor = state_goal_to_tensor(state_dict,self.device)
                a_idx,a = self.get_action(state_tensor)
                reward,done,new_state_dict = self.env.step(a)
                self.replay_buffer.push(state_dict,a_idx,reward,int(done),new_state_dict)
                state_dict=new_state_dict
                total_reward+=reward
                n_steps+=1
                
        return
    
    def train(self):
        #we start by filling the buffer memory with randomly generated data
        self.fill_memory()
        print("Replay Buffer filled")
        freq = max((self.n_episodes//5),1)
        for episode in range(self.n_episodes):
            start_time =time.time()
            state_dict=self.env.reset()
            done=False
            total_reward = 0
            n_steps = 0

            while(not done):
                #get q vector 
                state_tensor = state_goal_to_tensor(state_dict,self.device)
                state_tensor=state_tensor.to(device)#(device=self.device, dtype=torch.float)
                a_idx,a = self.get_action(state_tensor)

                reward,done,new_state_dict = self.env.step(a)
                #new (s,a) ->(r,d,next s) data is added to replay buffer     
                self.replay_buffer.push(state_dict,a_idx,reward,int(done),new_state_dict)

                total_reward+=reward
                #optimize the q and policy parameters
                q1_loss,q2_loss,policy_loss = self.optimize()
                n_steps+=1
                state_dict=new_state_dict

            self.total_rewards+=[total_reward]
            self.qlosses+=[(q1_loss+q2_loss)/2]
            self.policy_losses+=[policy_loss]
            self.steps+=[n_steps/self.env.max_steps]

            if ((episode+1)%self.update_interval==0):# update the target network at specified episodes
                self.update_target()
            total_time = time.time()-start_time
            if episode%freq==0:
                mean_reward = sum(self.total_rewards[-10:])/len(self.total_rewards[-10:])
                print('Episode ', episode, ': ', 'reward :',  mean_reward,
                      'loss:',q1_loss,q2_loss,'Policy loss:',policy_loss,'Episode Time:',total_time)
        return
    
    def evaluate(self,episodes,sample=1):
        total_rewards=[]
        steps=[]
        for episode in range(episodes):

            state_dict=self.env.reset()
            done=False
            total_reward = 0
            n_steps = 0

            while(not done):
                #get q vector 
                state_tensor = state_goal_to_tensor(state_dict,self.device)
                a_idx = torch.argmax(self.policy(state_tensor), dim=1, keepdim=True)# get action based on maximum probability in the policy
                a = self.env.action_list[a_idx]

                reward,done,new_state_dict = self.env.step(a)     

                total_reward+=reward
                n_steps+=1
                state_dict=new_state_dict

            total_rewards+=[total_reward]

            steps+=[n_steps/self.env.max_steps]

        avg_rewards = [sum(total_rewards[:idx+1][-sample:])/len(total_rewards[:idx+1][-sample:]) for idx in range(len(total_rewards))]
        avg_steps   = [sum(steps[:idx+1][-sample:])/len(steps[:idx+1][-sample:]) for idx in range(len(steps))]
        
        return avg_rewards,avg_steps
                
    def update_target(self):
        self.qtarget.load_state_dict(self.qn.state_dict())
        self.qtarget.eval()
        return
    
    def optimize(self):
        self.qn.train()
        self.policy.train()
        
        sg_batch,a_batch,r_batch,d_batch,nsg_batch = self.replay_buffer.sample(self.batch_size) 
        sg_batch.to(device=self.device, dtype=torch.float)
        a_batch.to(device=self.device, dtype=torch.float)
        r_batch.to(device=self.device, dtype=torch.float)
        d_batch.to(device=self.device, dtype=torch.float)
        nsg_batch.to(device=self.device, dtype=torch.float)
        max_reward = self.env.max_reward
        
        # Compute current q values

        q1,q2 = self.qn(sg_batch)
        qs = torch.min(q1.detach(),q2.detach()) # qs is computed from same network before the new update
        
        q1 = q1.gather(1, a_batch.long())
        q2 = q2.gather(1, a_batch.long())
        # Compute soft q targets
        
        p_nsg = self.policy(nsg_batch).detach()
        log_p_nsg = torch.log(p_nsg+1e-6)
        
        next_q1,next_q2=self.qtarget(nsg_batch)
        target_q_values = (p_nsg*(torch.min(next_q1.detach(),next_q2.detach())-
                                self.alpha*log_p_nsg)).sum(dim=1).unsqueeze(1)*self.gamma-(1-d_batch) + (r_batch/max_reward)

        # Compute q MSE Loss based on bellman equation
        loss_q1 = F.mse_loss(q1, target_q_values)
        loss_q2 = F.mse_loss(q2, target_q_values)

        #policy loss
        p = self.policy(sg_batch)
        log_p = torch.log(p+1e-6)                    
        loss_policy = (p*(self.alpha*log_p-qs)).mean()
        # Optimize the model
        self.qn1_optimizer.zero_grad()
        self.qn2_optimizer.zero_grad()
        loss_q1.backward()
        loss_q2.backward()
        
        self.qn1_optimizer.step()
        self.qn2_optimizer.step()
        
        #Optimize the policy
        self.policy_optimizer.zero_grad()
        loss_policy.backward()
            
        self.policy_optimizer.step()
        return loss_q1.item(),loss_q2.item(),loss_policy.item()
    
    def get_output(self,output='reward',is_avg=True,sample=1):
        data = []

        if(output=='steps'):
            data = self.steps
        elif(output=='q_loss'):
            data = self.qlosses
        elif(output=='policy_loss'):
            data = self.policy_losses
        else:
            data = self.total_rewards
            
        return [sum(data[:idx+1][-sample:])/len(data[:idx+1][-sample:]) for idx in range(len(data))]
            