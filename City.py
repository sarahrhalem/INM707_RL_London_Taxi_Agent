# -*- coding: utf-8 -*-
"""
@author: Sarah Rhalem

The implementation of the city environment to test tabular reinforcement learning and deep reinforcement learning algorithms

The environment models a city as a set of concentric circles. A fixed number of points are placed on the circles to models possible 
locations for the agent/driver. these locations are refered to in the code as stations

The driver can move from one location to another by either moving:
    - Moving across the zone (cw,acw): moving clockwise or anticlockwise on the same circle 
    - Moving across the road (in, out): moving to a near circle, in such move the same angle from the center remains the same.  


The architecture of environment is parameterized with number of road,zones and sations as well as the distance between each circle. 

Each road and zone will have a speed limit. The speed on the zones can be impacted by traffic jam, for simplicity, we assume that
traffic jam can occur at the same time in all zones with a given probaility. 

There are two type of roads, roads where traffic can be both ways and roads that are only one way. The type of the roads is considered
static and does not change when the environment is reset. 
 
The Agent is a car driver that moves from a starting station to a target station. We assume that the car dirver only starts at outer 
zone and the target station is the symmetrical wrt the center of the circles. 

The drive will recieve a reward equal to circumference of the outer circle divided by the minimum speed limit.

The driver can move from one station to another by choosing one of the actions described above. After each move the driver receives
a negative reward equal minus the time spent when moving between the two stations. This time is computed using the speed
limit on the zone/road that the driver selects. If a the driver selects a move across the zone and there is a traffic jam, 
we will then use half of the speed limit to compute the time spent between two stations. 

The drive will be penalized when moving in the wrong direction in a way, the penality will be receiving addition 10 times spent to 
move across the road. 

The driver will also be penalized when trying to move out of outer circle, the penality is computed as double of time needed to move
to a nearby station on outer circle.

If the driver reaches the maximum number of steps specified without reaching the target, it will be 
penalized based on the distance to target and minimum speed set in the environment 

Each move that driver makes will be counted as a step, the game will terminate after the driver reaches the target station or after 
a maximum number of steps is reached. 

"""

import matplotlib
import math
from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import numpy as np

##Station class to help computing the next station and the drivers moves

class Station:
    def __init__(self,radius_idx,angle_idx,n_zones,n_stations):
        self.radius_idx = radius_idx
        self.angle_idx  = angle_idx
        self.n_zones     = n_zones
        self.n_stations     = n_stations
        self.index = angle_idx+radius_idx*n_stations
        self.x = math.cos(2*math.pi*angle_idx/n_stations)
        self.y = math.sin(2*math.pi*angle_idx/n_stations)
 
    def __eq__(self,station):
        is_equal = (self.radius_idx==station.radius_idx)and(self.angle_idx==station.angle_idx)and(self.n_zones==station.n_zones)and(self.n_stations==station.n_stations)
        return is_equal
    def __str__(self):
        return str((self.radius_idx,self.angle_idx))
    
    def symetrical(self):
        return Station(self.radius_idx,(self.angle_idx+int(self.n_stations/2))%self.n_stations,self.n_zones,self.n_stations)

    def obs(self):
        return [(self.radius_idx+1)/self.n_zones,(self.angle_idx+1)/self.n_stations]        
    def next_station(self,direction):
        # we consider that the center of the zones is not part of the possible locations, so a move 'in' in the inner zone will be
        # a move the symmetrical point on same zone. 
        if direction =='in':
            if(self.radius_idx==0):
                return self.symetrical()
            else:
                return Station(self.radius_idx-1,self.angle_idx,self.n_zones,self.n_stations)
        elif direction == 'out':
            if(self.radius_idx<(self.n_zones-1)):
                return Station(self.radius_idx+1,self.angle_idx,self.n_zones,self.n_stations)
            else:
                return self
        elif direction == 'cw':#clockwise
            return Station(self.radius_idx,(self.angle_idx-1)%self.n_stations,self.n_zones,self.n_stations)
        elif direction == 'acw' :#anticlockwise
            return Station(self.radius_idx,(self.angle_idx+1)%self.n_stations,self.n_zones,self.n_stations)
        else:
            raise ValueError("Station:next_station: unvalid direction")
    def neighbours(self):
        directions =  ['in','out','cw','acw']
        neighbours_list = [self.next_station(direction) for direction in directions]
        try:
            s_index = neighbours_list.index(self)
            neighbours_list.remove(self)
            directions.remove(directions[s_index])
            return neighbours_list,directions
        except ValueError:
            return neighbours_list,directions
        
## Function to help with computing the distances between station
## The distance is computed as the shortest path to the target, the short test path between station1 
## and station 2 can be see as path on the city to reach the zone with smallest radius then move cw/acw
## on that zone to reach the other point.
 
def compute_distance(radius1,radius2,angle1,angle2,n_angles,is_straight=False):
    if((abs(angle1-angle2)==int(n_angles/2))and is_straight):
        return radius1+radius2
    else:
        distance_radius = abs(radius1-radius2)
        
        distance_angle = 2*math.pi*min((angle1-angle2)%n_angles,(angle2-angle1)%n_angles)*min(radius1,radius2)/n_angles
        
        return distance_radius+distance_angle

def neighbour_stations_idx(idx,n_zones,n_stations):
    station = Station(int(idx//n_stations),int(idx%n_stations),n_zones,n_stations)
    idx_list = [station.next_station(direction).index for direction in ['in','out']]
    try:
        idx_list.remove(idx)
        return idx_list
    except ValueError:
        return idx_list
    
class City:
    #l_radius[0] is of the radius of inner zone, the rest of element of l_radius are the distances betwene the successive  zones
    #oneway_roads is the percentage of one way road from he total number of roads
    
    def __init__(self,n_zones,l_radius,l_zones_speeds,l_road_speeds,n_stations=2,
                 p_trafficjam=None,oneway_roads=0,max_steps=None,n_obs=2,seed =0):
        
        self.n_zones = n_zones
        self.l_radius = l_radius
        
        self.zones_speed = l_zones_speeds
        self.road_speeds = l_road_speeds
        
        self.n_stations = n_stations
        self.action_list = ['acw','cw','in','out']
        
        self.p_trafficjam = p_trafficjam
        
        if (max_steps is None):
            self.max_steps = self.n_zones*self.n_stations
        else:
            self.max_steps = max_steps
        
        self.n_obs     = n_obs
        self.state_dim = int(3+2*(4**(n_obs+1)-1)/3) #the dimentionality of observables for deep RL algorithms
        
        self.n_states = self.n_stations**2*self.n_zones*2 #the number of states of environment, attribute needed for tabular RL
        
        total_radius = sum(self.l_radius)
        min_speed    = min(self.zones_speed+self.road_speeds)
        self.max_reward =  2*math.pi*total_radius/min_speed
    
        self.zone_state = 0
        self.rng   = np.random.default_rng(seed=seed)
        
        #state of environment
            # state of one way roads is static for each environment. it should only be part of the intialization
        
                # a matrix that will store the type of traffic in each road: 
                #self.directions[idx,jdx] is 0 for both ways,1 from station idx
                # to station jdx, and -1 if the traffic is from 
                
        self.oneway_roads = int(oneway_roads*(n_stations/2+(n_zones-1)*n_stations))
                                
        self.roads_direction = np.zeros((self.n_stations*self.n_zones,self.n_stations*self.n_zones))
        
        set_roads = set()
        for idx in range(self.n_stations*self.n_zones):
            for jdx in neighbour_stations_idx(idx,self.n_zones,self.n_stations):
                set_roads.add((idx,jdx))
        
        oneway_count = 0
        while((len(set_roads)>0) and (oneway_count<self.oneway_roads)):
            idx_pair = self.rng.choice(list(set_roads),1)[0]            
            self.roads_direction[idx_pair[0],idx_pair[1]]=1
            self.roads_direction[idx_pair[1],idx_pair[0]]=-1
            set_roads.remove((idx_pair[0],idx_pair[1]))
            set_roads.remove((idx_pair[1],idx_pair[0]))
            oneway_count+=1            
        
        self.reset()  
        self.current_loc = self.start_loc
        self.steps = 0
    
    def update(self):
        
        if(self.p_trafficjam is not None):
            self.zone_state = self.rng.choice([1,0],p=[self.p_trafficjam ,1-self.p_trafficjam ])
        else:
            self.zone_state = 0
            
        return
    def reset(self):
        # identify start and target location
        
        self.start_loc = Station(self.n_zones-1,self.rng.choice(range(self.n_stations)),self.n_zones,self.n_stations)
        
        self.target_loc = self.start_loc.symetrical()
            
        self.current_loc = self.start_loc
        self.steps = 0
        self.update()
        return self.step_output()
    
    def step(self,action):
        self.steps +=1
        reward,done =self.compute_reward(self.current_loc,action,self.target_loc)
        self.current_loc = self.current_loc.next_station(action)
        
        self.update()
        obs = self.step_output()
        return reward,done,obs
    
    def compute_reward(self,location,action,target):
        reward=0
        is_straight = action=='in' or action=='out'
        next_loc = location.next_station(action)
        distance = compute_distance(sum(self.l_radius[:location.radius_idx+1]),
                                    sum(self.l_radius[:next_loc.radius_idx+1]),
                                    location.angle_idx,next_loc.angle_idx,self.n_stations,is_straight)
        #for case distance =0 due to wrong direction on outer zone, we assume the agent will remain at
        # same locaiton and will be penalized for amount equivalent to moving on outer zone
        
        
        base_speed = 0
        ratio = 1
        if(action =='in'):
            base_speed = self.road_speeds[location.radius_idx]
        elif(action =='out'):
            base_speed = self.road_speeds[next_loc.radius_idx]
        else:
            base_speed = self.zones_speed[location.radius_idx]
            ratio      = (1-0.5*self.zone_state)
        
        d_time = distance/base_speed/ratio
        
        reward -=d_time
        #penality for driving in wrong direction
        if((is_straight) and (self.roads_direction[location.index,next_loc.index]==-1)):
            reward-=10*distance/base_speed
        
        #for case distance =0 due to wrong direction on outer zone, we assume the agent will remain at
        # same locaiton and will be penalized for amount equivalent to moving on outer zone
        if(distance==0):
            reward -=4*math.pi*sum(self.l_radius)/self.n_stations/base_speed
        done=False  
        if(next_loc==target):

            reward+=self.max_reward
            done=True
        elif (self.steps>= self.max_steps):
            #The agent is penalized by an amount equals to the distance to target when it fails to
            # reach distination
            dist_2_tgt = compute_distance(sum(self.l_radius[:next_loc.radius_idx+1]),
                                          sum(self.l_radius[:target.radius_idx+1]),
                                          next_loc.angle_idx,target.angle_idx,self.n_stations,True)
            min_speed = min(self.zones_speed+self.road_speeds)
            reward-= dist_2_tgt/min_speed
            done=True
        else:
            done=False
        return reward,done
    
    def step_output(self):
        #Method that computes the different outputs that are returned by the class to describe the state
        # state_idx is the index of the state as part of all possible states
        # target,current_loc,taffic: are more explicit describtion of the state of the environment
        # obs is the observation of the nearest neighbours, the state of the trafiic and the target 
        # location the each location is described with polar coordinates
        #[(radius index+1)/len(n_zones),(angle index+1)/len(n_stations)]
        tgt_angle = self.target_loc.angle_idx
        loc_idx   = self.current_loc.index
        state_idx = tgt_angle*self.n_stations*self.n_zones*2+loc_idx*2 + int(self.zone_state)
        #observation returned are current and target location with next possible locations in the next two moves 
        obs  = self.target_loc.obs()
        obs += generate_obs(self.current_loc,self.action_list,self.n_obs)
    
        obs+=[float(self.zone_state)+1e-6]
        state ={'state_idx':state_idx,'target':self.target_loc,
                'current_loc':self.current_loc,'taffic':self.zone_state,'observation':obs}
        return state
    
    def get_vis(self):
        circles =[]
        for idx in range(self.n_zones):
            circles+= [plt.Circle((0, 0), radius=sum(self.l_radius[:idx+1]), edgecolor='silver',facecolor='None',
                             label='zone circular',linestyle="--", linewidth=3)]
        roads = []
        r0 = self.l_radius[0]
        r  = sum(self.l_radius)
        for angle_idx in range(int(self.n_stations/2)):
            inner_station= Station(0,angle_idx,self.n_zones,self.n_stations)
            
            road = [[(-r*inner_station.x,-r*inner_station.y),(r*inner_station.x,r*inner_station.y)]]
            plt_rooad =mc.LineCollection(road, colors='g',label='roads',linestyle=":")
            roads+=[plt_rooad]
            
        start = (r*self.start_loc.x,r*self.start_loc.y)
        current_r = sum(self.l_radius[:self.current_loc.radius_idx+1])
        current = (current_r*self.current_loc.x,current_r*self.current_loc.y)
        target = (r*self.target_loc.x,r*self.target_loc.y)
        
        oneway_roads = []
        for angle_idx in range(self.n_stations):
            station  = Station(0,angle_idx,self.n_zones,self.n_stations)
            opposite = station.symetrical()
            if (self.roads_direction[station.index,opposite.index]==-1):
                oneway_roads+=[[(r0*opposite.x,r0*opposite.y),
                               (r0*station.x,r0*station.y)]]
                
                
            for zone_idx in range(1,self.n_zones):
                low_r = sum(self.l_radius[:zone_idx])
                high_r = sum(self.l_radius[:zone_idx+1])
                out_station = Station(zone_idx,angle_idx,self.n_zones,self.n_stations)
                in_station  = Station(zone_idx-1,angle_idx,self.n_zones,self.n_stations)
                
                if (self.roads_direction[out_station.index,in_station.index]==-1):
                    oneway_roads += [[(low_r*in_station.x,low_r*in_station.y),
                                      (high_r*out_station.x,high_r*out_station.y)]]
                elif(self.roads_direction[in_station.index,out_station.index]==-1):
                    oneway_roads += [[(high_r*out_station.x,high_r*out_station.y),
                                      (low_r*in_station.x,low_r*in_station.y)]]
                else:
                    continue
            
            
        outputs = {}
        outputs['circles'] = circles
        outputs['roads'] = roads
        outputs['current_loc'] = current
        outputs['target_loc'] = target
        outputs['start_loc'] = start
        outputs['n_zones']= self.n_zones
        outputs['max_radius'] = sum(self.l_radius)
        outputs['oneway_roads'] = oneway_roads
        return outputs
    def get_current_loc_cooridnates(self):
        r = sum(self.l_radius[:self.current_loc.radius_idx+1])
        res = (r*self.current_loc.x,r*self.current_loc.y)
        return res
def generate_obs(location,actions,n_obs):
    if n_obs==0:
        return location.obs()
    else:
        res = location.obs()
    
        for a in actions:
            res+=generate_obs(location.next_station(a),actions,n_obs-1)
        return res
        