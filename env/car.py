import pybullet as p
import os
import math
import numpy as np

class THH:
    # Simple Omnidirectional robot
    def __init__(self, client, baseOrientation=[0, 0, 0, 1], basePosition=[0, 0, 0.5], speed=5.0):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'thh.urdf')
        self.car = p.loadURDF(fileName=f_name,
                              baseOrientation=baseOrientation,
                              basePosition=basePosition,
                              physicsClientId=client)

        p.changeDynamics(self.car, -1, lateralFriction=0, spinningFriction=0, rollingFriction=0)

        self.speed = speed
        self.xy_speed = (0, 0)

    def get_ids(self):
        return self.car, self.client

    def apply_action(self, action):
        # Expects action to be two dimensional
        action[0], action[1] = np.clip(action[0], -1, 1), np.clip(action[1], -1, 1)
        norm = max(1, np.sqrt(action[0] ** 2 + action[1] ** 2))
        self.xy_speed = (action[0] * self.speed / norm, action[1] * self.speed / norm)
        p.resetBaseVelocity(objectUniqueId=self.car, linearVelocity=[self.xy_speed[0], self.xy_speed[1], 0])

    def get_position(self):
        # Get the position and orientation of the car in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.car, self.client)
        ang = p.getEulerFromQuaternion(ang)
        pos = pos[:2]

        # position
        observation = pos

        return observation

