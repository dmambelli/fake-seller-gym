import dataclasses
import datetime
from typing import List

import pandas as pd
import gym
from gym import spaces
import numpy as np


INFINITE_TIMESTEP = 1000
PAST_BINS = [-INFINITE_TIMESTEP, -28, -14, -7, -5, -3, -1, 0, 1]
PAST_AND_FUTURE_BINS = [-INFINITE_TIMESTEP, -28, -14, -7, -5, -3, -1, 0, 1, 3, 5, 7, 14, 28, INFINITE_TIMESTEP]



@dataclasses.dataclass
class Order:
    creation_timestamp: datetime.datetime
    delivery_date: datetime.date
    delivery_confirmation_timestamp: datetime.datetime


class FakeSellerGymEnv(gym.Env):

    def __init__(self, sellers_df: pd.DataFrame, discorvery_df: pd.DataFrame, orders_df: pd.DataFrame):
        self.signup_timestamp = None
        self.classifier_score = None
        self.orders: List[Order] = None
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
        # check for terminal state

        # fetch new events

        # update state with new events
        orders_vector = ...
        delivery_vector = ...
        timestep += 1
        obs = np.array([orders_vector, delivery_vector, timestep, self.classifier_score])

        # compute rewards
        reward = ...

        # update info
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
        creation = np.zeros(8) # -inf, -28, -14, -7, -5, -3, -1, 0 days
        delivery = np.zeros(15) # -inf, -28, -14, -7, -5, -3, -1, 0, 1, 3, 5, 7, 14, 28, inf days
        delivery_confirmation = np.zeros(8) # -inf, -28, -14, -7, -5, -3, -1, 0 days
        timestep = 0
        obs = np.array([creation, delivery, delivery_confirmation, timestep, self.classifier_score])

        info = {}
        return obs, info

    def _get_state_of_orders_at_timestep(self, timestep):
        if timestep == 0:
            creation = np.zeros(8) 
            delivery = np.zeros(14) 
            delivery_confirmation = np.zeros(8)
        else:
            creation_timesteps = np.array([(order.creation_timestamp.date() - self.signup_timestamp.date()).days for order in self.orders])
            creation, _ = np.histogram(creation_timesteps-timestep, bins=PAST_BINS)
            delivery_timesteps = np.array([(order.delivery_date - self.signup_timestamp.date()).days for order in self.orders])
            delivery_timesteps = delivery_timesteps[creation_timesteps < timestep] # Our actions happen at the end of a day 23:59 
            delivery, _ = np.histogram(delivery_timesteps-timestep, bins=PAST_AND_FUTURE_BINS)
            delivery_confirmation_timesteps = np.array([(order.delivery_confirmation_timestamp - self.signup_timestamp).days for order in self.orders])
            delivery_confirmation, _ = np.histogram(delivery_confirmation_timesteps-timestep, bins=PAST_BINS)
        return creation, delivery, delivery_confirmation

