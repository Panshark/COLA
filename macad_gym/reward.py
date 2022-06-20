from cv2 import exp
import numpy as np
import math


class Reward(object):
    def __init__(self):
        self.reward = 0.0
        self.prev = None
        self.curr = None
        self.stop = 0.0

    def compute_reward(self, prev_measurement, curr_measurement, flag, stop):
        self.prev = prev_measurement
        self.curr = curr_measurement
        self.stop = stop
        if flag == "corl2017":
            return self.compute_reward_corl2017()
        elif flag =="corl2022":
            return self.compute_reward_corl2022()
        elif flag == "lane_keep":
            return self.compute_reward_lane_keep()
        elif flag == "custom":
            return self.compute_reward_custom()

    def compute_reward_custom(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
        self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) / 10
        new_damage = (
            self.curr["collision_vehicles"] +
            self.curr["collision_pedestrians"] + self.curr["collision_other"] -
            self.prev["collision_vehicles"] -
            self.prev["collision_pedestrians"] - self.prev["collision_other"])
        if new_damage:
            self.reward -= 100.0

        self.reward -= self.curr["intersection_offroad"] * 0.05
        self.reward -= self.curr["intersection_otherlane"] * 0.05

        if self.curr["next_command"] == "REACH_GOAL":
            self.reward += 100

        return self.reward

    def compute_reward_corl2017(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        # Distance travelled toward the goal in m
        self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
        # Change in speed (km/h)
        self.reward += 0.05 * (
            self.curr["forward_speed"] - self.prev["forward_speed"])
        # New collision damage
        self.reward -= .00002 * (
            self.curr["collision_vehicles"] +
            self.curr["collision_pedestrians"] + self.curr["collision_other"] -
            self.prev["collision_vehicles"] -
            self.prev["collision_pedestrians"] - self.prev["collision_other"])

        # New sidewalk intersection
        self.reward -= 2 * (self.curr["intersection_offroad"] -
                            self.prev["intersection_offroad"])

        # New opposite lane intersection
        self.reward -= 2 * (self.curr["intersection_otherlane"] -
                            self.prev["intersection_otherlane"])

        return self.reward
    
    def compute_reward_corl2022(self):
        
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        # print(prev_dist - cur_dist)
        # Distance travelled toward the goal in m
        # self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
        # print("distance_to_goal", cur_dist)
        # Change in speed (km/hr)
        self.reward += 1 * (self.curr["forward_speed"] - self.prev["forward_speed"])
        # print("forward speed", self.curr["forward_speed"])
        # if abs(self.curr["forward_speed"] - self.prev["forward_speed"]) >= 10:
        #     self.reward -= 0.1 * (
        #                 abs(self.curr["forward_speed"] - self.prev["forward_speed"]) - 10) ** 2
        # print(self.curr["forward_speed"])
        if self.curr["forward_speed"] <= 30:
            self.reward += 1 *self.curr["forward_speed"] ** 2 / 9.0
        elif self.curr["forward_speed"] <= 50:
            self.reward += 1 * 5 * (50 - self.curr["forward_speed"])
        else:
            self.reward -= 1 * 2 * (self.curr["forward_speed"] - 50) ** 2

        if self.curr["forward_speed"] < 0:
            self.reward = self.reward - 0.005
            
        # # New collision damage
        # new_damage = (
        #     self.curr["collision_vehicles"] +
        #     self.curr["collision_pedestrians"] + self.curr["collision_other"] -
        #     self.prev["collision_vehicles"] -
        #     self.prev["collision_pedestrians"] - self.prev["collision_other"])
        # if new_damage:
        #     self.reward -= 100.0
        # # Sidewalk intersection
        # self.reward -= self.curr["intersection_offroad"]
        # # Opposite lane intersection
        # self.reward -= self.curr["intersection_otherlane"]
        
        if self.curr["forward_speed"] >= 0.50:
            self.stop = 0
        
        if self.curr["forward_speed"] < 0.50 and self.prev["forward_speed"] < 0.50:
            self.stop +=1
            try:
                temp_stop = math.exp(self.stop)
            except OverflowError:
                temp_stop = float('inf')
            self.reward = self.reward - 0.005*np.clip(0.0000001*temp_stop, 0.0, 50)
        
        # print("stop_setps", 0.1*np.clip(0.0000001*math.exp(self.stop), 0.0, 50), self.stop)
        # New collision damage
        new_damage = (
            self.curr["collision_vehicles"] +
            self.curr["collision_pedestrians"] + self.curr["collision_other"] -
            self.prev["collision_vehicles"] -
            self.prev["collision_pedestrians"] - self.prev["collision_other"])
        if new_damage:
            self.reward -= 100.0
            self.reward -= self.curr["max_steps"] - self.curr["step"]
            
        if self.curr["step"] == self.curr["max_steps"]:
            self.reward += 100

        # # New sidewalk intersection
        # self.reward -= 15 * (self.curr["intersection_offroad"] -
        #                     self.prev["intersection_offroad"])

        # # New opposite lane intersection
        # self.reward -= 15 * (self.curr["intersection_otherlane"] -
        #                     self.prev["intersection_otherlane"])
        
        # Sidewalk intersection
        self.reward -= 0.1 * self.curr["intersection_offroad"]
        # Opposite lane intersection
        self.reward -= 0.1 * self.curr["intersection_otherlane"]

        return self.reward, self.stop

    def compute_reward_lane_keep(self):
        self.reward = 0.0
        # Speed reward, up 30.0 (km/h)
        self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) / 10
        # New collision damage
        new_damage = (
            self.curr["collision_vehicles"] +
            self.curr["collision_pedestrians"] + self.curr["collision_other"] -
            self.prev["collision_vehicles"] -
            self.prev["collision_pedestrians"] - self.prev["collision_other"])
        if new_damage:
            self.reward -= 100.0
        # Sidewalk intersection
        self.reward -= self.curr["intersection_offroad"]
        # Opposite lane intersection
        self.reward -= self.curr["intersection_otherlane"]

        return self.reward

    def destory(self):
        pass
