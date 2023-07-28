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
    
    def u2v(self, ur, ul):
        v = 0.5*self.r*(ur+ul)
        omega = (self.r/self.L)*(ur-ul)
        return v, omega
    
    def reset(self):
        if self.simOn:
            self.sim.stopSimulation()
            while self.sim.getSimulationState() != self.sim.simulation_stopped:
                pass
        loc = self.sim.getObjectPosition(self.spawn[np.random.randint(0,6)], -1)
        self.sim.setObjectPosition(self.robot, -1, loc)
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
        self.sim.setJointTargetVelocity(self.motorL, 0.0)
        self.sim.setJointTargetVelocity(self.motorL1, 0.0)
        
        self.sim.setJointTargetVelocity(self.motorR, 0.0)
        self.sim.setJointTargetVelocity(self.motorR1, 0.0)

    def step(self):
        res = np.zeros((4,))
        dist = np.zeros((4,))
        for k in range(4):
            res[k], dist[k], _, _, _ =self.sim.readProximitySensor(self.sensor[k])
        #if not resR and not resL:
        #    return 0
        dsig = np.multiply(dist, 1*res)
        lgain = np.linspace(-1, -.1, 4)
        rgain = np.linspace(-.1, -1, 4)
        ul = 0.5+np.dot(dsig, lgain)
        ur = 0.5+np.dot(dsig, rgain)
        T = self.sim.getSimulationTime()
        while self.sim.getSimulationTime() - T < 0.1:
            self.sim.setJointTargetVelocity(self.motorL, ul)
            self.sim.setJointTargetVelocity(self.motorL1, ul)

            self.sim.setJointTargetVelocity(self.motorR, ur)
            self.sim.setJointTargetVelocity(self.motorR1, ur)

            self.client.step()
        self.titer += 1
        v, omega = self.u2v(ur, ul)
        self.tdist += v*0.5
        nxt_state = self.getobs()
        done = False
        resV, distV, _, _, _ = self.sim.readProximitySensor(self.vsensor)
        #print(resV)
        #print(distV)
        if resV > 0 or self.titer > self.maxiter:
            done = True
        if resV:
            reward = -100
        info = {'name': 'pioneer3-AT', 'iter': self.titer, 'dist': self.tdist}
        reward = 1
        return nxt_state, reward, done, info

    def __del__(self):
        if self.simOn:
            self.sim.stopSimulation()

env = Pioneer()

dist = []
iterc = []
for i in range(100):
    state = env.reset()
    done = False
    loc = [env.sim.getObjectPosition(env.robot, -1)]
    while not done:
        next_state, reward, done, info = env.step() 
        state = next_state
        loc.append(env.sim.getObjectPosition(env.robot, -1))

    aloc = np.array(loc)
    dist.append(np.linalg.norm(aloc[-1,:]-aloc[0,:]))
    iterc.append(info['iter'])

np.savetxt('braitdist.txt', np.array(dist))
np.savetxt('braititer.txt', np.array(iterc))