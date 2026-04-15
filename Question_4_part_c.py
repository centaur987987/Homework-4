# Andrew Centa
# c) **(20 points)** Write a function which generates 1000 samples by sampling next states from $P(s'|s)$, then sampling observations from $P(y, a|s)$.

from Question_4_Part_b import P_y_a_s, P_ss
import numpy as np

def generate_samples(P_ss, P_y_a_s, samples = 1000, start_state = 0):
    states = []
    observations = []

    s = start_state

    for i in range(samples):
        states.append(s)
        
        probs = P_y_a_s[:, s]
        probs = probs / probs.sum()

        obs_index = np.random.choice(16, p = probs)
        observations.append(obs_index)

        s =  np.random.choice(12, p = P_ss[s])

        return np.array(states), np.array(observations)

states, observations = generate_samples(P_ss, P_y_a_s, samples = 1000, start_state = 0)

print(states)

print(observations)