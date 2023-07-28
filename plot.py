"""
Obstacle avoidance for a mobile robot using Q-learning
Plotting script

Authors: 
M.A. Rojas Andrade
L. R. García Vázquez
E. Y. Aguilera Camacho
M.A. Escobar Carmona
A.S. Torres Villegas
J. P. Ramírez Paredes <jpi.ramirez@ugto.mx>
University of Guanajuato (2023)
"""

import matplotlib.pyplot as plt
import numpy as np

rw = np.loadtxt('RLhreward.txt')
rwavg = np.convolve(rw, 0.1*np.ones((10,)), mode='same')

RLdist = np.loadtxt('RLdist.txt')
Bdist = np.loadtxt('braitdist.txt')

print(np.mean(RLdist))
print(np.sqrt(np.var(RLdist)))
print(np.mean(Bdist))
print(np.sqrt(np.var(Bdist)))

plt.plot(rw)
plt.plot(rwavg)
plt.ylabel('Recompensa acumulada')
plt.xlabel('Episodios')
plt.show()