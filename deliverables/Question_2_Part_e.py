# Andrew Centa

# execute temporal difference learning
# I commented a completed print at the bottom.

from reinforcement_learning4e import PassiveTDAgent, run_single_trial
from mdp4e import GridMDP, best_policy
import random

# print grid cleanly
def gridmdp_str(self):
    rows = []
    for row in reversed(self.grid):  # reversed so row 0 is at the bottom
        rows.append(' | '.join(
            f'{cell:>6}' if cell is not None else '  None' 
            for cell in row
        ))
    return '\n'.join(rows) + f'\n(gamma={self.gamma})\n'

GridMDP.__str__ = gridmdp_str

# learning costs
cost = [-0.1, -0.08, -0.04, -0.02, -0.001]

# create grid
def create_grid(step_cost):
    return GridMDP([[step_cost, step_cost, step_cost, +1],
                    [step_cost, None, step_cost, -1],
                    [step_cost, step_cost, step_cost, step_cost]],
                   terminals=[(3, 2), (3, 1)],
                   gamma=0.99)

for c in cost:
    mdp = create_grid(c)

    
    # initialize actions policy
    pi = {s: mdp.actions(s)[0] for s in mdp.states}

    # initialize the passive TDA agent
    agent = PassiveTDAgent(pi, mdp)

    # train
    for i in range(10000):
        run_single_trial(agent, mdp)

    # enact the best policy from the 10000 trials
    learned_policy = best_policy(mdp, agent.U)

    # print the step cost
    print(f"\nStep cost: {c}")

    # print as arrows using the to_arrows function
    print(mdp.to_arrows(learned_policy))


    ## Sample print
    '''Step cost: -0.1
[['>', '>', '>', '.'], ['^', None, '^', '.'], ['^', '<', '^', '^']]

Step cost: -0.08
[['>', '>', '>', '.'], ['^', None, '^', '.'], ['^', '<', '^', '^']]

Step cost: -0.04
[['>', '>', '^', '.'], ['^', None, '^', '.'], ['^', '<', '^', '<']]

Step cost: -0.02
[['^', '>', '^', '.'], ['^', None, '^', '.'], ['^', '<', '^', '<']]

Step cost: -0.001
[['>', '>', '^', '.'], ['^', None, '^', '.'], ['^', '<', '<', 'v']]'''