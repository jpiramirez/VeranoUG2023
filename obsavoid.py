"""
Obstacle avoidance for a mobile robot using Q-learning
This RL implementation uses CoppeliaSim with the ZeroMQ API.

Author: Juan-Pablo Ramirez-Paredes <jpi.ramirez@ugto.mx>
University of Guanajuato (2023)
"""

import time
import math as m

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import numpy as np


class pioneer():
    def __init__(self):
        self.r = 0.5*0.195
        self.L = 2.0*0.1655
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.motorL = self.sim.getObject('/PioneerP3DX/leftMotor')
        self.motorR = self.sim.getObject('/PioneerP3DX/rightMotor')
        self.sensorL = self.sim.getObject('/PioneerP3DX/sensorL')
        self.sensorR = self.sim.getObject('/PioneerP3DX/sensorR')
        self.robot  = self.sim.getObject('/PioneerP3DX')
        self.vsensor = self.sim.getObject('/PioneerP3DX/vsensor')
        self.client.setStepping(True)
        self.simOn = False
        self.obsn = 7
        self.actn = 4
        self.detthresh = 0.25
        self.titer = 0
        self.maxiter = 100
    
    def v2u(self, v, omega):
        ur = v/self.r + self.L*omega/(2*self.r)
        ul = v/self.r - self.L*omega/(2*self.r)
        return ur, ul
    
    def reset(self):
        if self.simOn:
            self.sim.stopSimulation()
            while self.sim.getSimulationState() != self.sim.simulation_stopped:
                pass
        self.sim.startSimulation()
        self.titer = 0
        self.simOn = True
        return 0
    
    def getobs(self):
        resL, distL, _, _, _ =self.sim.readProximitySensor(self.sensorL)
        resR, distR, _, _, _ =self.sim.readProximitySensor(self.sensorR)
        #if not resR and not resL:
        #    return 0
        if resR and distR < self.detthresh and not resL:
            return 1
        if resL and distL < self.detthresh and not resR:
            return 2
        if resR and distR < self.detthresh and resL and distL < self.detthresh:
            return 3
        if resR and distR > self.detthresh and not resL:
            return 4
        if resL and distL > self.detthresh and not resR:
            return 5
        if resR and distR > self.detthresh and resL and distL > self.detthresh:
            return 6
        return 0
    
    def move(self, dir):
        if dir == 0:
            ur, ul = self.v2u(0.2, 0.0)
        elif dir == 1:
            ur, ul = self.v2u(-0.2, 0.0)
        elif dir == 2:
            ur, ul = self.v2u(0.2, np.pi/2)
        elif dir == 3:
            ur, ul = self.v2u(0.2, -np.pi/2)
        else:
            ur = 0
            ul = 0
        T = self.sim.getSimulationTime()
        while self.sim.getSimulationTime() - T < 1.0:
            self.sim.setJointTargetVelocity(self.motorL, ul)
            self.sim.setJointTargetVelocity(self.motorR, ur)
            self.client.step()
        self.sim.setJointTargetVelocity(self.motorL, 0.0)
        self.sim.setJointTargetVelocity(self.motorR, 0.0)
    
    def step(self, action):
        self.move(action)
        self.titer += 1
        if action == 0:
            reward = 10
        elif action == 1:
            reward = -10
        else:
            reward = 5
        nxt_state = self.getobs()
        done = False
        resV, distV, _, _, _ = self.sim.readProximitySensor(self.vsensor)
        if resV > 0 or self.titer > self.maxiter:
            done = True
        if resV:
            reward = -100
        info = {'name': 'pioneerP3Dx', 'iter': self.titer}
        return nxt_state, reward, done, info

    def __del__(self):
        if self.simOn:
            self.sim.stopSimulation()

env = pioneer()

q_table = np.zeros([env.obsn, env.actn])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

hreward = []

for i in range(1, 1000):
    state = env.reset()

    ereward = []
    done = False
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, 4) 
        else:
            action = np.argmax(q_table[state]) 

        next_state, reward, done, info = env.step(action) 
 
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state

        ereward.append(reward)
    
    hreward.append(np.sum(ereward))
        
    if i % 100 == 0:
        print(f"Episode: {i}")
        print(q_table)


print("Training finished.\n")
env.sim.stopSimulation()

print(repr(q_table))

np.savetxt('RLhreward.txt', np.array(hreward))
