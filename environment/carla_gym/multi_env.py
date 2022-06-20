#!/usr/bin/env python
import time

from macad_gym.carla.multi_env import MultiCarlaEnv

import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Carlamacad(MultiCarlaEnv):
  def __init__(self, Config = "environment/carla_gym/config.json"):
        self.configs = json.load(open(Config))
        self._spec = lambda: None
        self._spec.id = "Carla-v0"
        self._singelID = list(self.configs["actors"].keys())[0]
        super(Carlamacad, self).__init__(self.configs)
        
        self._observation_space = self.observation_space[self._singelID]
        self.observation_space = self._observation_space
        self._action_space = self.action_space[self._singelID]
        self.action_space = self._action_space
        # print(self.observation_space.shape)
  
  def reset(self):
      obs = super(Carlamacad, self).reset()
      
      return obs[self._singelID]
  
  def step(self, action):
      action_dict = {}
      action_dict[self._singelID] = int(action)
    #   print("action",action_dict)
      obs, reward, done, info, weather_num = super(Carlamacad, self).step(action_dict)
      return obs[self._singelID], reward[self._singelID], done[self._singelID], info[self._singelID], weather_num
  
  
def get_next_actions(measurements, is_discrete_actions):
        """Get/Update next action, work with way_point based planner.

        Args:
            measurements (dict): measurement data.
            is_discrete_actions (bool): whether use discrete actions

        Returns:
            dict: action_dict, dict of len-two integer lists.
        """
        action_dict = {}
        for actor_id, meas in measurements.items():
            m = meas
            command = m["next_command"]
            if command == "REACH_GOAL":
                action_dict[actor_id] = 0
            elif command == "GO_STRAIGHT":
                action_dict[actor_id] = 3
            elif command == "TURN_RIGHT":
                action_dict[actor_id] = 6
            elif command == "TURN_LEFT":
                action_dict[actor_id] = 5
            elif command == "LANE_FOLLOW":
                action_dict[actor_id] = 3
            # Test for discrete actions:
            if not is_discrete_actions:
                action_dict[actor_id] = [1, 0]
        return action_dict
       

if __name__ == "__main__":
    
    

    env = Carlamacad()
    multi_env_config = env.configs
    
    for _ in range(2):
        obs = env.reset()

        total_reward_dict = {}
        action_dict = {}

        env_config = multi_env_config["env"]
        actor_configs = multi_env_config["actors"]
        for actor_id in actor_configs.keys():
            total_reward_dict[actor_id] = 0
            if env._discrete_actions:
                action_dict[actor_id] = 3  # Forward
            else:
                action_dict[actor_id] = [1, 0]  # test values

        start = time.time()
        i = 0
        done = False
        while not done:
            # while i < 20:  # TEST
            i += 1
            obs, reward, done, info, _ = env.step(action_dict[actor_id])
            action_dict = get_next_actions(info, env._discrete_actions)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward
            print(":{}\n\t".join(["Step#", "rew", "ep_rew",
                                  "done {}"]).format(i, reward,
                                                    total_reward_dict, done))

        print("{} fps".format(i / (time.time() - start)))
