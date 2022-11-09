import numpy as np
import pandas as pd
from simulator.environment import Order
import dataclasses
import datetime
from random import shuffle
import string
import random
ts = datetime.datetime.fromisoformat


def id_generator(size=8, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

@dataclasses.dataclass
class Seller:
    seller_id: int
    classifier_score: float
    discovery_timestamp: datetime.date
    signup_timestamp: datetime.datetime
    orders: list[Order]

class SyntheticDataGen():
    def __init__(self, p_fraud_seller, beta_lower_cs, beta_higher_cs, exponential_creation, exponential_delivery, exponential_delivery_confirmation):
        self.p_fraud_seller = p_fraud_seller
        self.beta_lower = beta_lower_cs
        self.beta_higher = beta_higher_cs
        self.exponential_creation = exponential_creation
        self.exponential_delivery = exponential_delivery
        self.exponential_delivery_confirmation = exponential_delivery_confirmation
        self.max_orders_per_seller = 10
        self.init_datetime = ts("2023-01-01 00:00:00")
        self.cutoff_day = 90
    
    def get_dataset(self, num_sellers: int):
        # create df
        sellers_df = pd.DataFrame(columns=['seller_id', 'signup_timestamp', 'classifier_score'])
        discovery_df = pd.DataFrame(columns=['seller_id', 'discovery_timestamp'])
        orders_df = pd.DataFrame(columns=['seller_id', 'creation_timestamp', 'delivery_date', 'delivery_confirmation_timestamp'])
        
        orders_count = 0
        fraud_perc = 0
        # sample sellers
        for idx in range(num_sellers):
            seller = self.sample_seller()
            
            sellers_df.loc[idx] = [seller.seller_id, seller.signup_timestamp, seller.classifier_score]
            discovery_df.loc[idx] = [seller.seller_id, seller.discovery_timestamp]
            if seller.discovery_timestamp is not None:
                fraud_perc += 1
            
            for order in seller.orders:
                orders_df.loc[orders_count] = [seller.seller_id, order.creation_timestamp, order.delivery_date, order.delivery_confirmation_timestamp]
                orders_count += 1
                
        print(f'% seller fraud: {fraud_perc/num_sellers}')
        
        return sellers_df, discovery_df, orders_df
        
    def sample_seller(self):
        seller_id = id_generator()
        # sample random time in the year after init_datetime
        signup_timestamp = self.init_datetime + datetime.timedelta(0, int(np.random.choice(60*60*24*365, 1)[0]))
        # sample fraud
        fraud = np.random.binomial(1, self.p_fraud_seller)
        # sample classifier score
        if fraud:
            classifier_score = np.random.beta(self.beta_higher, self.beta_lower)
            #NOTE: discovery timestamp is selected uniformly across the time horizon
            discovery_timestamp = signup_timestamp + datetime.timedelta(0, int(np.random.choice(60*60*24*self.cutoff_day, 1)[0]))
        else:
            classifier_score = np.random.beta(self.beta_lower, self.beta_higher)
            discovery_timestamp = None
            
        seller = Seller(
            seller_id=seller_id,
            classifier_score=classifier_score,
            discovery_timestamp=discovery_timestamp,
            signup_timestamp=signup_timestamp,
            orders=None)
            
        # sample #orders
        num_orders = int(np.random.choice(np.arange(1, self.max_orders_per_seller), 1)[0])
        # sample orders
        orders = []
        prev_event = signup_timestamp
        for _ in range(num_orders):
            orders.append(self.sample_orders_from_seller(seller, prev_event))
            prev_event = orders[-1].creation_timestamp

        #NOTE: probably not necessary
        shuffle(orders)
        
        seller.orders = orders
            
        return seller
            
    def sample_orders_from_seller(self, seller: Seller, prev_event: datetime.datetime) -> Order:
        order = Order(creation_timestamp=None,delivery_date=None,delivery_confirmation_timestamp=None)
        time_horizon_timestamp = seller.signup_timestamp+datetime.timedelta(0, 60*60*24*self.cutoff_day)
        terminal_timestamp = seller.discovery_timestamp if seller.discovery_timestamp is not None else time_horizon_timestamp
        # sample creation as offset from signup
        creation_timestamp = prev_event + datetime.timedelta(0, int(np.random.exponential(self.exponential_creation)))
        # making sure we sampled something in the horizon
        while creation_timestamp > terminal_timestamp:
            creation_timestamp = prev_event + datetime.timedelta(0, int(np.random.exponential(self.exponential_creation)))
        order.creation_timestamp = creation_timestamp
        
        # sample delivery as offset from creation
        delivery_date = (order.creation_timestamp + datetime.timedelta(0, int(np.random.exponential(self.exponential_delivery)))).date()
        order.delivery_date = delivery_date # delivery date always exists since we get it when the order is made
        
        # sample delivery confirmation as offeset from delivery
        if seller.discovery_timestamp is None and order.delivery_date is not None: # if not fraud and there is a delivery data
            delivery_datetime = datetime.datetime.combine(order.delivery_date, datetime.time(0,0,0))
            delivery_confirmation_timestamp = delivery_datetime + datetime.timedelta(0, int(np.random.exponential(self.exponential_delivery_confirmation)))
            if delivery_confirmation_timestamp > terminal_timestamp:
                order.delivery_confirmation_timestamp = None
            else:
                order.delivery_confirmation_timestamp = delivery_confirmation_timestamp
        else:
            order.delivery_confirmation_timestamp = None
        return order
        
    