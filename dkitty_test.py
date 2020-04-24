import robel
import gym
import math

if __name__ == "__main__":
        env = gym.make('DKittyWalkFixed-v0')
        done = False
        ang60 = math.pi/3
        while True:
                env.reset()
                done = False
                while True:
                        a = env.action_space.sample()
                        ss, r, done, info = env.step(a)
                        env.render()