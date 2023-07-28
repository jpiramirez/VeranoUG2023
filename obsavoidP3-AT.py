"""
Obstacle avoidance for a mobile robot using Q-learning
This RL implementation uses CoppeliaSim with the ZeroMQ API.

Author: Miguel Angel Rojas Andrade <ma.rojasandrade@ugto.mx>
University of Guanajuato (2023)
"""

from P3AT4A16O import *

env = Pioneer()

q_table = np.zeros([env.obsn, env.actn])
q_table_count = np.zeros([env.obsn, env.actn])

Neps = 1000
# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1
epsilon_decay_rate = epsilon/Neps

hreward = []

for i in range(1, Neps):
    state = env.reset()
    ereward = []
    done = False
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, 6) 
        else:
            action = np.argmax(q_table[state]) 

        next_state, reward, done, info = env.step(action) 
 
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        q_table_count[state, action] += 1
        state = next_state
    
        ereward.append(reward)
    
    epsilon = max(epsilon - epsilon_decay_rate, 0)
    print(f'Episode {i}  Epsilon={epsilon}')

    hreward.append(np.sum(ereward))
        
    if i % 100 == 0:
        print(f"Episode: {i}")
        print(q_table)


print("Training finished.\n")
env.sim.stopSimulation()

print(repr(q_table))

np.savetxt('RLhreward.txt', np.array(hreward))
np.savetxt('q_table.txt', q_table)
np.savetxt('q_table_count.txt', q_table_count)