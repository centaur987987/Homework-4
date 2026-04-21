# Andrew Centa
# c) **(20 points)** Write a function which generates 1000 samples by sampling next states from $P(s'|s)$, then sampling observations from $P(y, a|s)$.

from Question_4_Part_b import P_y_a_s, P_ss
import numpy as np

def generate_samples(P_ss, P_y_a_s, samples = 1000, start_state = 0):
   
    # initialize the states and observations and start space
    states = []
    observations = []
    s = start_state

    # cycle for 1000 samples
    for i in range(samples):
        states.append(s) # append states by s
        
        probs = P_y_a_s[:, s] # find columns of Pyas for state s
        probs = probs / probs.sum() # normalize

        obs_index = np.random.choice(16, p = probs) # randomly pick one of the observations
        observations.append(obs_index) # append the observation deliverable to include the new index

        s =  np.random.choice(12, p = P_ss[s]) # randomly select a new state

    return np.array(states), np.array(observations)

states, observations = generate_samples(P_ss, P_y_a_s, samples = 1000, start_state = 0)

print('Question 4c')
print('Next States')
print(states)
print('\n Observations')
print(observations)