import envs
import gym     

if __name__ == "__main__":
    env = gym.make('Humanoid_Motion-v0')
    state = env.reset()
    print(state)
    print(env.action_space)
    print(env.action_space.sample())
    while True:
        state, reward, done, forceSensor = env.step([0]*6)
    env.close()