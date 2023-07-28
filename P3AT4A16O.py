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

import time
import math as m

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import numpy as np


class Pioneer():
    def __init__(self):
        self.r = 0.5*0.195
        self.L = 2.0*0.1655
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.motorL = self.sim.getObject('/PioneerP3AT/left_motor')
        self.motorL1 = self.sim.getObject('/PioneerP3AT/left_motor1')

        self.motorR = self.sim.getObject('/PioneerP3AT/right_motor')
        self.motorR1 = self.sim.getObject('/PioneerP3AT/right_motor1')

        self.robot  = self.sim.getObject('/PioneerP3AT')

        self.sensor = []
        self.sensor.append(self.sim.getObject('/PioneerP3AT/sensor[0]'))
        self.sensor.append(self.sim.getObject('/PioneerP3AT/sensor[1]'))
        self.sensor.append(self.sim.getObject('/PioneerP3AT/sensor[2]'))
        self.sensor.append(self.sim.getObject('/PioneerP3AT/sensor[3]'))
        self.vsensor = self.sim.getObject('/PioneerP3AT/vsensor')

        self.spawn = []
        for k in range(6):
            self.spawn.append(self.sim.getObject(f'/obs[{k}]'))

        self.client.setStepping(True)
        self.simOn = False
        self.obsn = 16
        self.actn = 6
        self.detthresh = 0.4
        self.titer = 0
        self.tdist = 0
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
        loc = self.sim.getObjectPosition(self.spawn[np.random.randint(0,6)], -1)
        loc[0] += np.random.normal(0, 0.1)
        loc[1] += np.random.normal(0, 0.1)
        self.sim.setObjectPosition(self.robot, -1, loc)
        self.sim.setObjectOrientation(self.robot, -1, [0, 0, np.pi*np.random.randint(0,365)/180.0])
        self.sim.startSimulation()
        self.titer = 0
        self.tdist = 0
        self.simOn = True
        return 0
    
    def getobs(self):
        res = np.zeros((4,))
        for k in range(4):
            res[k], _, _, _, _ =self.sim.readProximitySensor(self.sensor[k])
        #if not resR and not resL:
        #    return 0
        val = int(res[3]*8 + res[2]*4 + res[1]*2 + res[0]*1)
        return val
    
    def move(self, dir):
        if dir == 0:
            ur, ul = self.v2u(0.1, 0.0)
        elif dir == 1:
            ur, ul = self.v2u(-0.1, 0.0)
        elif dir == 2:
            ur, ul = self.v2u(0.1, np.pi/4)
        elif dir == 3:
            ur, ul = self.v2u(0.1, -np.pi/4)
        elif dir == 4:
            ur, ul = self.v2u(0.1, np.pi/2)
        elif dir == 5:
            ur, ul = self.v2u(0.1, -np.pi/2)
        else:
            ur = 0
            ul = 0
        T = self.sim.getSimulationTime()
        while self.sim.getSimulationTime() - T < 0.1:
            self.sim.setJointTargetVelocity(self.motorL, ul)
            self.sim.setJointTargetVelocity(self.motorL1, ul)

            self.sim.setJointTargetVelocity(self.motorR, ur)
            self.sim.setJointTargetVelocity(self.motorR1, ur)

            self.client.step()
        
        if dir == 1:
            self.tdist += -0.1*0.1
        else:
            self.tdist += 0.1*0.1

    def step(self, action):
        self.move(action)
        self.titer += 1
        if action == 0:
            reward = 20
        elif action == 1:
            reward = -10
        else:
            reward = 5
        nxt_state = self.getobs()
        done = False
        resV, distV, _, _, _ = self.sim.readProximitySensor(self.vsensor)
        #print(resV)
        #print(distV)
        if resV or self.titer >= self.maxiter:
            done = True
        if resV:
            reward = -100
        info = {'name': 'pioneer3-AT', 'iter': self.titer, 'dist': self.tdist}
        return nxt_state, reward, done, info

    def __del__(self):
        if self.simOn:
            self.sim.stopSimulation()
