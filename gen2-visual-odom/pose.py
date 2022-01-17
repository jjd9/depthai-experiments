#!/usr/bin/env python3
import numpy as np

class Pose:
    def __init__(self):
        self.position = np.array([0,0,0], dtype=float) # cartesian position
        # TODO: Make this a quaternion
        self.orientation = np.eye(3, dtype=float) # euler orientation
    
    def translate(self, translation):
        self.position += translation.reshape(3,)
    
    def rotate(self, rotation):
        self.orientation = np.dot(rotation, self.orientation) 