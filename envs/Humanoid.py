from types import CoroutineType
import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces

class humanoid(gym.Env):
    def __init__(self) -> None:
        super(humanoid, self).__init__()
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-45, cameraPitch=-20, cameraTargetPosition=[0,0,0.1])
        self.reset()
    
    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        p.setJointMotorControlArray(self.dancerUid, [self.joint2Index[joint] for joint in self.joint_names], p.POSITION_CONTROL, action)
        p.stepSimulation()
        state, _, sensorState = self.getObservation()
        return state, 0, False, sensorState

    def getObservation(self):
        jointStates = {}
        for joint in self.joint_names:
            jointStates[joint] = p.getJointState(self.dancerUid, self.joint2Index[joint])
        
        # check collision and get sensor position
        collision = False
        for link in self.link_names:
            if len(p.getContactPoints(bodyA=self.dancerUid, linkIndexA=self.link2Index[link])) > 0:
                collision = True
                for contact in p.getContactPoints(bodyA=self.dancerUid, bodyB=self.dancerUid, linkIndexA=self.link2Index[link]):
                    print("Collision Occurred in Link {} & Link {}!!!".format(contact[3], contact[4]))
                    p.changeVisualShape(self.dancerUid, contact[3], rgbaColor=[1,0,0,1])
                    p.changeVisualShape(self.dancerUid, contact[4], rgbaColor=[1,0,0,1])
        
        # check sensor
        sensorStates = {}
        for sensor in self.sensor_name:
            sensorStates[sensor[0]] = (p.getJointState(self.dancerUid, self.sensor2Index[sensor[0]])[2], p.getLinkState(self.dancerUid, self.link2Index[sensor[1]])[0])
        
        observation = [jointStates[joint][0] for joint in self.joint_names]
        self.get_zmp(sensorStates)
        return observation, collision, sensorStates
    
    def reset(self):
        p.resetSimulation()
        self.step_counter = 0
        self.dancerUid = p.loadURDF("./half_human.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]),
            flags=p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8) 
        self.groudId = p.loadURDF("plane.urdf")
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(1./60.)

        self.joint_names = []
        self.joint2Index = {} # index map to jointName
        self.link_names = []
        self.link2Index = {} # index map to linkName
        self.lower_limits = []
        self.upper_limits = []
        self.init_angles = []
        self.sensor_name = []
        self.sensor2Index = {}

        for index in range(p.getNumJoints(self.dancerUid)):
            jointName = p.getJointInfo(self.dancerUid, index)[1].decode('utf-8')
            linkName = p.getJointInfo(self.dancerUid, index)[12].decode('utf-8')
            self.link_names.append(linkName)
            self.link2Index[linkName] = index
            if jointName == 'joint_wolrd':
                continue
            if 'sensor' in jointName:
                self.sensor_name.append((jointName, linkName))
                self.sensor2Index[jointName] = index
                p.enableJointForceTorqueSensor(self.dancerUid, index, enableSensor=True)
                continue
            self.joint_names.append(jointName)
            self.joint2Index[jointName] = index
            self.lower_limits.append(-np.pi)
            self.upper_limits.append(np.pi)
            self.init_angles.append(0)
        self.action_space = spaces.Box(np.array(self.lower_limits,dtype=np.float), np.array(self.upper_limits,dtype=np.float))
        self.observation_space = spaces.Box(np.array(self.lower_limits,dtype=np.float), np.array(self.upper_limits,dtype=np.float))

        state, _, sensorState = self.getObservation()
        return state
    
    def close(self):
        p.disconnect()
    
    def get_zmp(self, sensorState):
        sigma_PFx = 0
        sigma_PFy = 0
        sigma_F = 0
        for key, value in sensorState.items():
            F, pos = value
            if pos[2] > 0.025:
                continue
            sigma_PFx += pos[0] * F[2]
            sigma_PFy += pos[1] * F[2]
            sigma_F += F[2]
        px = sigma_PFx/(sigma_F + 1e-8)
        py = sigma_PFy/(sigma_F + 1e-8)
        line_length = 0.2
        p.addUserDebugLine([px + line_length/2.0, py, 0], [px - line_length/2.0, py, 0], lineColorRGB=[1,0,0], lineWidth=4, lifeTime=1/120.0)
        p.addUserDebugLine([px, py + line_length/2.0, 0], [px, py - line_length/2.0, 0], lineColorRGB=[1,0,0], lineWidth=4, lifeTime=1/120.0)
        return (px, py, 0)