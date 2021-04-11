# INM707_RL_London_Taxi_Agent
## A custom RL environment designed to simulate a taxi agent navigating challenges and zones in London

- City.py has the implemntation for the city environment with step and reward calcualtion
- BaseAgent.py has the implementation of the an agent that decides on an action based on the policy provided, this module includes few basic policies to test City environment
- Visualisation.py has a matplotlib function to visualise the environment and trajectories taken by agent
- QLAgent.py an implementation of the agent that acts on optimal policy based on Q Learning algorithm
- SACD.py an implementation of an agent that trains based on Soft Actor-Critic algorithm
- ultis.py utility module to save and load SACD class

### SAC Results are stored within the results folder and can be loaded using the util.load_results method
