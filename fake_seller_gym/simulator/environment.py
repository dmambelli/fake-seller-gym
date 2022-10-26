import enum
import dataclasses
import datetime
from this import d
from typing import List, Optional

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


class Action(enum.Enum):
    ALLOW = 0
    INVESTIGATE = 1
    

class FakeSellerGymEnv(gym.Env):

    def __init__(self, sellers_df: pd.DataFrame, discorvery_df: pd.DataFrame, orders_df: pd.DataFrame, final_timestep_if_genuine: int = 90):
        self.signup_timestamp = None
        self.classifier_score = None
        self.orders: List[Order] = None
        self.discovery_timestamp = None
        self.timestep = None
        self.discovery_timestep = None
        self.final_timestep_if_genuine = final_timestep_if_genuine
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

    def step(self, action: Action):
        self.timestep += 1
        
        # check for terminal state
        if (self.timestep == self.final_timestep) or (action==Action.INVESTIGATE):
            done = True
            
        # fetch new events

        # update state with new events
        orders_vector = ...
        delivery_vector = ...
        obs = np.array([orders_vector, delivery_vector, self.timestep, self.classifier_score])

        # compute rewards
        reward = ...

        # update info
        info = {}

        return obs, reward, done, info

    def reset(self):
        # sampling from dataset
        seller = self.sellers_df.sample(n=1).iloc[0]

        # init properties 
        self.signup_timestamp = seller.signup_timestamp
        self.classifier_score = seller.classifier_score
        self.orders = self._get_orders_for_seller_id(seller.seller_id)
        self.discovery_timestamp = self._get_discovery_timestamp_for_seller_id(seller.seller_id)
        # self.discovery_timestep = (self.discovery_timestamp - self.signup_timestamp).days + 1
        self.final_timestep = (self.discovery_timestamp - self.signup_timestamp).days + 1 if self.discovery_timestamp is not None else self.final_timestep_if_genuine
        

        # build observation
        self.timestep = 0
        creation, delivery, delivery_confirmation = self._get_state_of_orders_at_timestep(self.timestep)
        obs = np.concatenate((creation, delivery, delivery_confirmation, np.array([self.timestep, self.classifier_score])), axis=None)

        info = {}
        return obs, info
    
    def _get_reward_from_state_action(self, action, creation, delivery_confirmation):
        pass
        

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

    def _get_orders_for_seller_id(self, seller_id: int) -> List[Order]:
        return [
            Order(order.creation_timestamp, order.delivery_date, order.delivery_confirmation_timestamp) 
            for _, order in self.orders_df[self.orders_df.seller_id == seller_id].iterrows()
        ]

    def _get_discovery_timestamp_for_seller_id(self, seller_id: int) -> Optional[datetime.datetime]:
        discovery = self.discovery_df[self.discovery_df.seller_id==seller_id]
        if not discovery.empty:
            return discovery.iloc[0].discovery_timestamp
