import dataclasses
import datetime

import pandas as pd
import gym
from gym import spaces
import numpy as np


@dataclasses.dataclass
class Order:
    creation_timestamp: datetime.datetime
    delivery_timestamp: datetime.datetime


class FakeSellerGymEnv(gym.Env):

    def __init__(self, sellers_df: pd.DataFrame, discorvery_df: pd.DataFrame, orders_df: pd.DataFrame):
        self.signup_timestamp = None
        self.classifier_score = None
        self.orders = None
        self.discovery_timestamp = None
        # load the dataset
        self.sellers_df = sellers_df  # seller_id, signup_timestamp, classifier_score
        self.discovery_df = discorvery_df  # seller_id, discovery_timestamp
        self.orders_df = orders_df  # seller_id, creation_timestamp, delivery_timestamp

    @property
    def environment(self):
        return self._environment

    @property
    def game_over(self):
      return False

    @property
    def action_space(self):
        return action_space

    @property
    def observation_space(self):
 
        return obs_space

    def step(self, action):
    
        info = {}
        return obs, reward, done, info

    def reset(self):
        # sampling from dataset
        seller_id = ...

        # init properties 
        self.signup_timestamp = ...
        self.classifier_score = ...
        self.orders = ...
        self.discovery_timestamp = ...

        # build observation
        orders_vector = np.zeros(15)
        delivery_vector = np.zeros(15)
        timestep = 0
        obs = np.array([orders_vector, delivery_vector, timestep, self.discovery_timestamp])

        info = {}
        return obs, info
