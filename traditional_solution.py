import pinocchio
from walking_solver import solver
import numpy as np

if __name__ == "__main__":
    generator = solver()
    generator.compute_mass_core(np.array([0]*6))
    generator.forward_kinematic(np.array([np.pi/4.0, -np.pi/2.0, np.pi/4.0, 0, 0, 0]))
    target_T = pinocchio.SE3(np.eye(3), np.array([0.0, -0.20, 0.25]))
    Joint_name = 'ankle_left'
    q, success, err = generator.inverse_kinematic(Joint_name, target_T)
    print(q, success)

    generator.two_dimension_inverted_pendulum()