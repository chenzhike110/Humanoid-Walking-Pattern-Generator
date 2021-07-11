from os import name
import pinocchio
from pinocchio.utils import npToTuple
import numpy as np
from numpy.linalg import norm, solve

class solver(object):
    def __init__(self) -> None:
        super().__init__()
        self._model = pinocchio.buildModelFromUrdf("half_human.urdf")
        self._data = self._model.createData()
    
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
