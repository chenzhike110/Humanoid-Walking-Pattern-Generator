import numpy as np
from numpy.linalg import norm, solve

import pinocchio

class robanRobot(object):
    def __init__(self) -> None:
        self.model, self.collision_model, self.visual_model = pinocchio.buildModelsFromUrdf("./Roban.urdf")
        self.data = self.model.createData()
        self.q = pinocchio.randomConfiguration(self.model)

if __name__ == "__main__":
    robot = robanRobot()
