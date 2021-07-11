import pinocchio
from walking_solver import solver
import numpy as np

if __name__ == "__main__":
    generator = solver()
    generator.forward_kinematic(np.array([0]*6))
    target_T = pinocchio.SE3(np.eye(3), np.array([0.05, -0.20, 0.15]))
    Joint_name = 'ankle_right'
    q, success, err = generator.inverse_kinematic(Joint_name, target_T)
    print(q, success)