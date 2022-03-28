# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialising state and defining action space and state space"""
        self.action_space = [(p,q) for p in range(m) for q in range(m) if p !=q] + [(0,0)] 
        self.state_space = [(xi,ti,di) for xi in range(m) for ti in range(t) for di in range(d)]
        self.state_init = random.choice(self.state_space)
        self.time_elapsed = 0
        self.terminal = False
        self.reset()


        
    ## Encodinging the state for NN input:
    def state_encod_arch1(self, state):
        """converting the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. The vector is of size m + t + d."""
        state_encod = np.zeros((m+t+d,), dtype=int)
        state_encod[state[0]] = 1
        state_encod[m + state[1]] = 1
        state_encod[m + t + state[2]] = 1
        return state_encod

    

    # Use this function if you are using architecture-2 
#     def state_encod_arch2(self, state, action):
#         """converting the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
#         state_encod = np.zeros((m+t+d+m+m,), dtype=int)
#         state_encod[state[0]] = 1
#         state_encod[m + state[1]] = 1
#         state_encod[m + t +state[2]] = 1
#         state_encod[m + t + d + action[0]] = 1
#         state_encod[m + t + d + m + action[1]] = 1
#         return state_encod



    ## Getting requests basis location:
    def requests(self, state):
        """In a general scenario, the number of requests the cab driver can get at any state is not the same. We can model the number of requests as follows: The number of requests (possible actions) at a state is dependent on the location.For each location, we can sample the number of requests from a Poisson distribution using the mean λ defined for each location. Also,the upper limit on these customers’ requests is 15."""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range((m-1)*m), requests) 
        actions = [self.action_space[i] for i in possible_actions_index]
        actions.append((0,0)) # Apending no-ride’ action
        possible_actions_index.append(20)   # Apending index corresponding to no-ride’ action
        return possible_actions_index,actions   


    # Getting reward by taking a action from given state on the basis of Time_matrix:
    # Takes in state, action and Time-matrix and returns the reward
    def reward_func(self, state, action, Time_matrix):
        """we have assumed that both the cost and the revenue are purely functions of time, i.e. for every hour of driving, the cost (of battery and other costs) and the revenue (from the customer) is the same - irrespective of the traffic conditions, speed of the car etc. So, the reward function will be (revenue earned from pickup point 𝑝 to drop point 𝑞) - (Cost of battery used in moving from pickup point 𝑝 to drop point 𝑞) - (Cost of battery used in moving from current point 𝑖 to pick-up point 𝑝)."""
        #t1 = time required to reach from current location to pick-up location
        #p_time = current time at pick up location
        #p_day = current day at pick up location
        #t2 = time required to reach from pick-up location to drop location
        
        if action != (0,0): # if driver decides to take a request:
            if state[0] == action[0]:                               #if current location = pick-up location:
                t1 = 0                  
            else:                                                   #if current location != pick-up location
                t1 = int(Time_matrix[state[0],action[0],state[1],state[2]]) #time calculated from time matrix
            if state[1] + t1 > 23:                       #if day changes while reaching the pick-up location:
                p_time = (state[1] + t1) % 24
                day_delta = (state[1] + t1) // 24
                p_day = (state[2] + day_delta) % 7
            else:                                       #if reaching the pick-up location on the same day:
                p_time = state[1] + t1
                p_day = state[2]        
            t2 = int(Time_matrix[action[0], action[1], p_time, p_day]) #time calculated from time matrix
            reward = (R * t2) - (C * (t1 + t2))
        
        else:                                           # if driver decides to take a break:
            reward = -C
        
        return reward

    
    # Function for getting the next state as per the action taken in current state using Time matrix:
    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        # t1 = time required to reach from current location to pick-up location
        # p_time = current time at pick up location
        # p_day = current day at pick up location
        # t2 = time required to reach from pick-up location to drop location
        # q_time = current time at drop location
        # q_day = current day at drop location
        # state ---> (current location, current time, current day)
        # action ---> (pick-up location, drop location)
        
        if action != (0,0):                   # if driver decides to take a request:
            if (state[0] == action[0]):       #if current location = pick-up location:
                t1 = 0
            else:                             #if current location != pick-up location:
                t1 = int(Time_matrix[state[0],action[0],state[1],state[2]])  #time calculated from time matrix
            if state[1] + t1 > 23:            #if day changes while reaching the pick-up location
                p_time = (state[1] + t1) % 24
                day_delta = (state[1] + t1) // 24
                p_day = (state[2] + day_delta) % 7
            else:                             #if reaching the pick-up location on the same day
                p_time = state[1] + t1
                p_day = state[2]

            t2 = int(Time_matrix[action[0], action[1], p_time, p_day])
            if p_time + t2 > 23:              #if day changes while reaching the drop location
                q_time = (p_time + t2) % 24
                day_delta = (p_time + t2) // 24
                q_day = (p_day + day_delta) % 7
            else:                            #if reaching the drop location on the same day that of pickup
                q_time = p_time + t2
                q_day = p_day
            next_state = (action[1],q_time,q_day)   # Next state
            self.time_elapsed += t1 + t2     # total time elapsed reaching from current locn to drop locn
            
        else:                                # if driver decides to take a break:
            if state[1] + 1 > 23:            # if day changes after taking a break
                q_time = 0                   # Incrementing by 1 hr
                if state[2] + 1 > 6:         # if week changes after taking a break
                    q_day = 0                # adjusting the day as per new week
                else:                        # if week remains the same after taking a break
                    q_day = state[2]+1       # Incrementing by 1 day
            else:                            # if day doesnt changes after taking a break
                q_time = state[1] + 1        # Incrementing by 1 hr
                q_day = state[2]             # day will remain the same
            next_state = (state[0],q_time,q_day)  # Next state
            self.time_elapsed += 1
        
        if self.time_elapsed >= 720:       # if 30 days i.e. 30*24 hrs are over
            self.terminal = True           # episode is ended \ terminal state is reached.
        
        return next_state,self.terminal


    # for reseting the attributes:
    def reset(self):
        return self.state_space, self.action_space, self.state_init
