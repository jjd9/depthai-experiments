#!/usr/bin/env python3
import numpy as np

# Source https://learnopencv.com/rotation-matrix-to-euler-angles/
# Checks if a matrix is a valid rotation matrix.


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).


def rotationMatrixToEulerAngles(R):

    assert(isRotationMatrix(R))

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


class Pose:
    def __init__(self):
        self.position = np.array([0, 0, 0], dtype=float)  # cartesian position
        # TODO: Make this a quaternion
        self.orientation = np.eye(3, dtype=float)  # euler orientation

    def update(self, pose_update):
        old_pose = self.asHomogeneous()
        new_pose = np.dot(old_pose, np.linalg.inv(pose_update))
        self.position = new_pose[:3, 3].reshape(3,)
        self.orientation = new_pose[:3, :3]

    def asHomogeneous(self):
        pose_mat = np.eye(4, dtype=float)
        pose_mat[:3, 3] = self.position.reshape(3,)
        pose_mat[:3, :3] = self.orientation
        return pose_mat

    def getRPYString(self):
        rpy = rotationMatrixToEulerAngles(self.orientation)
        rpy *= 180 / np.pi
        text = "Roll: {}deg, Pitch: {}deg, Yaw: {}deg".format(
            *rpy.round())
        return text

    def getPositionString(self):
        text = "X: {}mm, Y: {}mm, Z: {}mm".format(*self.position.round())
        return text
