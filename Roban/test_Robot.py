import pybullet as p
import pybullet_data
import time

if __name__ == "__main__":
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-45, cameraPitch=-20, cameraTargetPosition=[0,0,0.1])
    p.resetSimulation()
    robotUid = p.loadURDF("./Roban.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]),
            flags=p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.8) 
    groudId = p.loadURDF("plane.urdf")
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(1./60.)

    while True:
        time.sleep(0.01)