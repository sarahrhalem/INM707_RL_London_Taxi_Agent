# Util functions:

import SACD
import City
import torch
import pickle

def save_checkpoint(state, checkpoint_path):
    print("Saving checkpoint ... ")
    torch.save(state, checkpoint_path)
    print("Checkpoint:", checkpoint_path, "saved.")


def load_checkpoint(model, optimizer, scheduler, load_checkpoint_path):
    print("Loading checkpoint ... ")
    checkpoint = torch.load(load_checkpoint_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, scheduler, start_epoch

def save_agent(agent, filename):
    rewards = agent.total_rewards
    steps = agent.steps
    results = {'rewards':rewards,'steps':steps}

    results['n_zones']=agent.env.n_zones
    results['l_radius']=agent.env.l_radius
    results['l_zones_speeds']=agent.env.zones_speed
    results['l_road_speeds']=agent.env.road_speeds
    results['n_stations']=agent.env.n_stations
    results['p_trafficjam']=agent.env.p_trafficjam
    results['max_steps']=agent.env.max_steps
    results['n_obs']=agent.env.n_obs
    results['roads_direction']=agent.env.roads_direction
    
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    torch.save(agent.policy.state_dict(),"policy_"+filename)
    return


def load_agent(filename):

    with open(filename, 'rb') as handle:
        results = pickle.load(handle)
        n_obs = results['n_obs'] if ('n_obs' in results) else 2
        env = City.City(n_zones=results['n_zones'],l_radius=results['l_radius'],
                        l_zones_speeds=results['l_zones_speeds'],l_road_speeds=results['l_road_speeds'],
                        n_stations=results['n_stations'],p_trafficjam=results['p_trafficjam'],
                        oneway_roads=0,max_steps=results['max_steps'],n_obs=n_obs)
        env.roads_direction = results['roads_direction']
        agent=SACD.SACD_Agent(env)
        agent.total_rewards = results['rewards']
        agent.steps = results['steps']
    
    checkpoint = torch.load("policy_"+filename)
    agent.policy.load_state_dict(checkpoint)
    return agent
