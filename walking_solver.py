from os import name
import pinocchio
import numpy as np
from numpy.linalg import norm, solve

class solver(object):
    def __init__(self) -> None:
        super().__init__()
        self._model_visiual = pinocchio.buildModelFromUrdf("half_human.urdf",pinocchio.JointModelFreeFlyer())
        self._data_visiual = self._model_visiual.createData()
        for name in self._model_visiual.frames:
            print(name)
        self._model = pinocchio.buildModelFromUrdf("half_human.urdf")
        self._data = self._model.createData()
    
    def generate_mask(self):
        mask = []
        for name in self._model_visiual.names:
            if name in self._model.names:
                mask.append(1)
            else:
                mask.append(0)
        self._mask = np.array(mask)
    
    def generate_q_visiual(self, q):
        q_visiual = np.zeros(len(self._mask))
        for i in range(len(self._mask)):
            if self._mask[i] > 0:
                q_visiual[i] = q[sum(self._mask[:i])]
        return q_visiual
    
    def forward_kinematic(self, q):
        # quat = pinocchio.Quaternion(np.random.rand(4,1)).normalized()
        # R = quat.toRotationMatrix()
        pinocchio.forwardKinematics(self._model, self._data, q)
        for name, oMi in zip(self._model.names, self._data.oMi):
            print(("{:<24} : {: .2f} {: .2f} {: .2f}, {: .2f} {: .2f} {: .2f}"
                .format( name, *oMi.translation.T.flat, *(pinocchio.rpy.matrixToRpy(oMi.rotation)).flat)))

    def getJointID(self, joint_name):
        for index, name in enumerate(self._model.names):
            if name == joint_name:
                return index
        print("joint not found!!!")
        return 0

    def inverse_kinematic(self, joint_name:str, state:pinocchio.SE3, IT_MAX:int=1000, eps:float=1e-4, DT:float=1e-1, damp:float=1e-6) -> list:
        Joint_ID = self.getJointID(joint_name)
        q = pinocchio.neutral(self._model)
        iteration = 0
        while True:
            pinocchio.forwardKinematics(self._model, self._data, q)
            dMi = state.actInv(self._data.oMi[Joint_ID])
            err = pinocchio.log(dMi).vector
            if norm(err) < eps:
                success = True
                break
            if iteration > IT_MAX:
                success = False
                break
            J = pinocchio.computeJointJacobian(self._model, self._data, q, Joint_ID)
            v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pinocchio.integrate(self._model, q, v*DT)
            iteration += 1
        return  q.flatten().tolist(), success, err.T
    
    def math_inverse_kinematic(self, joint_name, state:pinocchio.SE3):
        pass
    
    def compute_mass_core(self, q):
        centerofmass = pinocchio.centerOfMass(self._model, self._data, q)
        print(centerofmass)
        return centerofmass
    
    def two_dimension_inverted_pendulum(self):
        pinocchio.forwardKinematics(self._model, self._data, np.array([0]*6))
        max_length = pinocchio.centerOfMass(self._model, self._data)[2]
        initial_state = np.array([np.pi/4.0, -np.pi/2.0, np.pi/4.0, np.pi/4.0, -np.pi/2.0, np.pi/4.0])
        print(max_length)
        state_now = initial_state
        E1 = 0.1
        E2 = 0.1
        pinocchio.forwardKinematics(self._model, self._data, initial_state)
        # move left leg first
        stick_point = self._data.oMi[self.getJointID("ankle_right")].translation.T
        centerofmass = pinocchio.centerOfMass(self._model, self._data)
        z = centerofmass[2] - stick_point[2] + 0.1
        xf = np.math.sqrt(max_length**2 - z**2)
        x = centerofmass[0] - stick_point[0]
        Tc = np.math.sqrt(z/9.8)
        print(stick_point, centerofmass)

        vf = np.math.sqrt(9.8/z*xf**2+2*E1)
        tf = Tc*np.log((xf + Tc*vf)/x)
        print(tf)

        