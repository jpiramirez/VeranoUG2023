"""
Obstacle avoidance for a mobile robot using Q-learning
This RL implementation uses CoppeliaSim with the ZeroMQ API.

Authors: 
M.A. Rojas Andrade
L. R. García Vázquez
E. Y. Aguilera Camacho
M.A. Escobar Carmona
A.S. Torres Villegas
J. P. Ramírez Paredes <jpi.ramirez@ugto.mx>
University of Guanajuato (2023)
"""

from P3AT4A16O import *

env = Pioneer()
q_table = np.loadtxt('q_table.txt')
print(q_table)

dist = []
iterc = []


for i in range(100):
    state = env.reset()
    done = False
    loc = [env.sim.getObjectPosition(env.robot, -1)]
    while not done:
        action = np.argmax(q_table[state])
        #print(f'{state} {action} {q_table[state]}')
        next_state, reward, done, info = env.step(action) 
        state = next_state
        loc.append(env.sim.getObjectPosition(env.robot, -1))

    aloc = np.array(loc)
    dist.append(np.linalg.norm(aloc[-1,:]-aloc[0,:]))
    iterc.append(info['iter'])

np.savetxt('RLdist.txt', np.array(dist))
np.savetxt('RLiter.txt', np.array(iterc))