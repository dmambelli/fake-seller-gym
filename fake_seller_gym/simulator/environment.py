import enum
import dataclasses
import datetime
from typing import List, Optional

import pandas as pd
import gym
import numpy as np


INFINITE_TIMESTEP = 1000
epsilon = 1e-8
PAST_BINS = [-INFINITE_TIMESTEP, -28, -14, -7, -5, -3, -1, 0-epsilon]
PAST_AND_FUTURE_BINS = [-INFINITE_TIMESTEP, -28, -14, -7, -5, -3, -1, 0, 1, 3, 5, 7, 14, 28, INFINITE_TIMESTEP]
COST_INVESTIGATION = 2
AVERAGE_BENEFIT_PER_ORDER = 3
AVERAGE_COST_PER_ORDER = 4


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
        # Initialisation variables         
        self.final_timestep_if_genuine = final_timestep_if_genuine
        self.sellers_df = sellers_df  # seller_id, signup_timestamp, classifier_score
        self.discovery_df = discorvery_df  # seller_id, discovery_timestamp
        self.orders_df = orders_df  # seller_id, creation_timestamp, delivery_timestamp
        
        # Variables set in the .reset() method
        self.signup_timestamp = None
        self.classifier_score = None
        self.orders: List[Order] = None
        self.discovery_timestamp = None
        self.timestep = None
        self.final_timestep = None

    @property
    def environment(self):
        return self._environment

    @property
    def game_over(self):
      return False

    def step(self, action: Action):
        self.timestep += 1
        
        # check for terminal state
        done = (self.timestep == self.final_timestep) or (action==Action.INVESTIGATE)
                
        # if we investigate at timestep t and the seller is fraudulent, then all orders after timestep t are cancelled
        if action==Action.INVESTIGATE and self.discovery_timestamp is not None:
            self.orders = [order for order in self.orders if _get_timestep_from_timestamp(self.signup_timestamp, order.creation_timestamp) < self.timestep-1]
        
        creation, delivery, delivery_confirmation = _get_state_of_orders_at_timestep(self.signup_timestamp, self.orders, self.timestep)
        obs = np.concatenate((creation, delivery, delivery_confirmation, np.array([self.timestep, self.classifier_score])), axis=None)


        # compute rewards
        reward = self._get_reward_from_state_action(action, creation, delivery_confirmation, done)

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
        self.final_timestep = (self.discovery_timestamp - self.signup_timestamp).days + 1 if self.discovery_timestamp is not None else self.final_timestep_if_genuine


        # build observation
        self.timestep = 0
        creation, delivery, delivery_confirmation = _get_state_of_orders_at_timestep(self.signup_timestamp, self.orders, self.timestep)
        obs = np.concatenate((creation, delivery, delivery_confirmation, np.array([self.timestep, self.classifier_score])), axis=None)

        info = {}
        return obs, info
    
    def _get_reward_from_state_action(self, action, creation, delivery_confirmation, done):
        reward = 0 if self.discovery_timestamp is not None else AVERAGE_BENEFIT_PER_ORDER*delivery_confirmation[-1] # benefit of confirmation, never happening for a fraud seller
        
        if action==Action.INVESTIGATE:
            reward -= COST_INVESTIGATION # price of investigation
            if self.discovery_timestamp is None:
                done = False
                while not done:
                    _, reward_t, done, _ = self.step(action=Action.ALLOW)
                    reward += reward_t
                    
        if done and self.discovery_timestamp is not None: # Terminal cost
            reward -= AVERAGE_COST_PER_ORDER*np.sum(creation) # we make costs for every order
        return reward
        

    def _get_orders_for_seller_id(self, seller_id: int) -> List[Order]:
        return [
            Order(order.creation_timestamp, order.delivery_date, order.delivery_confirmation_timestamp) 
            for _, order in self.orders_df[self.orders_df.seller_id == seller_id].iterrows()
        ]

    def _get_discovery_timestamp_for_seller_id(self, seller_id: int) -> Optional[datetime.datetime]:
        discovery = self.discovery_df[self.discovery_df.seller_id==seller_id]
        if not discovery.empty:
            return discovery.iloc[0].discovery_timestamp




def _get_state_of_orders_at_timestep(signup_timestamp: datetime.datetime, orders: List[Order], timestep: int):
    if timestep == 0:
        creation = np.zeros(7) 
        delivery = np.zeros(14) 
        delivery_confirmation = np.zeros(7)
    else:
        #creation_timesteps = np.array([(order.creation_timestamp.date() - signup_timestamp.date()).days for order in orders])
        creation_timesteps = np.array([_get_timestep_from_timestamp(signup_timestamp, order.creation_timestamp) for order in orders])
        creation, _ = np.histogram(creation_timesteps-timestep, bins=PAST_BINS)
        delivery_timesteps = np.array([(order.delivery_date - signup_timestamp.date()).days for order in orders])
        delivery_timesteps = delivery_timesteps[creation_timesteps < timestep] # Our actions happen at the beginning of the day 00:01 
        delivery, _ = np.histogram(delivery_timesteps-timestep, bins=PAST_AND_FUTURE_BINS)
        #delivery_confirmation_timesteps = np.array([(order.delivery_confirmation_timestamp - signup_timestamp).days for order in orders])
        delivery_confirmation_timesteps = np.array([_get_timestep_from_timestamp(signup_timestamp, order.delivery_confirmation_timestamp) for order in orders])
        delivery_confirmation, _ = np.histogram(delivery_confirmation_timesteps-timestep, bins=PAST_BINS)
    return creation, delivery, delivery_confirmation


def _get_timestep_from_timestamp(signup_timestamp: datetime.datetime, timestamp: datetime.datetime) -> int:
    return (timestamp.date() - signup_timestamp.date()).days