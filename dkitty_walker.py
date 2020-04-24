import robel
import gym
import math
import matplotlib.pyplot as plt
import os

ANG30 = math.radians(30)
ANG45 = math.radians(45)
ANG60 = math.radians(60)
ANG90 = math.radians(90)

class DKittyWalker():
    def __init__(self):
        self._g1 = {
            '1': [0.0, 0.0, 0.0],
            '2': [ANG30, -ANG60, ANG30],
            '3': [ANG30, -ANG30, ANG60],
            '4': [0.0, 0.0, 0.0]
        }
        self._g2 = {
            '1': [-ANG60, 0.0, 0.0],
            '2': [ANG30, -ANG60, ANG30],
            '3': [ANG30, -ANG30, ANG60],
            '4': [0.0, 0.0, 0.0]
        }
        self._g3 = {
            '1': [0.0, -ANG60, ANG60],
            '2': [ANG60, -ANG60, ANG60],
            '3': [ANG60, -ANG60, ANG60],
            '4': [0.0, -ANG60, ANG60]
        }
        self._g4 = {
            '1': [0.0, -ANG60, ANG60],
            '2': [ANG60, -ANG60, ANG60],
            '3': [ANG60, -ANG60, ANG60],
            '4': [0.0, -ANG60, ANG60]
        }
        self._g5 = {
            '1': [0.0, -ANG60, ANG60],
            '2': [ANG60, -ANG60, ANG60],
            '3': [ANG60, -ANG60, ANG60],
            '4': [0.0, -ANG60, ANG60]
        }
        self._g6 = {
            '1': [0.0, -ANG60, ANG60],
            '2': [ANG60, -ANG60, ANG60],
            '3': [ANG60, -ANG60, ANG60],
            '4': [0.0, -ANG60, ANG60]
        }
        self._g7 = {
            '1': [0.0, -ANG60, ANG60],
            '2': [ANG60, -ANG60, ANG60],
            '3': [ANG60, -ANG60, ANG60],
            '4': [0.0, -ANG60, ANG60]
        }
        self._g8 = {
            '1': [0.0, -ANG60, ANG60],
            '2': [ANG60, -ANG60, ANG60],
            '3': [ANG60, -ANG60, ANG60],
            '4': [0.0, -ANG60, ANG60]
        }
        self._env = gym.make('DKittyWalkRandomDynamics-v0')

    def get_action_arr(self, joint_pos):
        leg1 = joint_pos['1']
        leg2 = joint_pos['2']
        leg3 = joint_pos['3']
        leg4 = joint_pos['4']
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        action[0] = leg1[0]
        action[1] = leg1[1]
        action[2] = leg1[2]
        action[3] = leg2[0]
        action[4] = leg2[1]
        action[5] = leg2[2]
        action[6] = leg3[0]
        action[7] = leg3[1]
        action[8] = leg3[2]
        action[9] = leg4[0]
        action[10] = leg4[1]
        action[11] = leg4[2]
        return action

    def run(self):
        dist_to_goal = []
        init_state = self._env.reset()
        g1 = {
            '1': [0.0, 0.0, -ANG60],
            '2': [0.0, 0.0, 0.0],
            '3': [0.0, 0.0, 0.0],
            '4': [0.0, 0.0, 0.0]
        }
        g2 = {
            '1': [0.0, 0.0, 0.0],
            '2': [0.0, 0.0, -ANG60],
            '3': [0.0, 0.0, 0.0],
            '4': [0.0, 0.0, 0.0]
        }
        g3 = {
            '1': [0.0, 0.0, 0.0],
            '2': [0.0, 0.0, 0.0],
            '3': [0.0, 0.0, -ANG60],
            '4': [0.0, 0.0, 0.0]
        }
        g4 = {
            '1': [0.0, 0.0, 0.0],
            '2': [0.0, 0.0, 0.0],
            '3': [0.0, 0.0, 0.0],
            '4': [0.0, 0.0, -ANG60]
        }
        positions = [g1, g2, g3, g4]
        index = 0
        counter = 0
        overall_count = 0;
        dist_to_goal.append(init_state[-1])
        while True:
            print(overall_count)
            # curr_action = stable
            # curr_action[index] -= math.radians(1.0)
            ss, r, done, info = self._env.step(self.get_action_arr(positions[index]))
            dist_to_goal.append(ss[-1])
            self._env.render()
            counter += 1
            overall_count += 1
            if (counter // 2) == 4:
                counter = 0
                index = 0
            else:
                index = counter // 2
            if overall_count == 500:
                break

        # Save plots of performance metrics
        if not os.path.exists('plots'):
            os.makedirs('plots')
        # plt.rcParams['figure.figsize'] = [20, 6]
        plt.plot(range(len(dist_to_goal)), dist_to_goal, 'r')
        plt.ylabel('Distance to Goal Location')
        plt.xlabel('Timestep')
        plt.title('Distance to Goal Location for Non-RL Walking Algorithm (Random Env)')
        plt.savefig('plots/dist_to_goal_non_rl_random.png')


if __name__ == "__main__":
    agent = DKittyWalker()
    agent.run()