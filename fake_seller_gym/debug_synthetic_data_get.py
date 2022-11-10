from simulator.environment import FakeSellerGymEnv
from synthetic_data_gen.generator import SyntheticDataGen
import statistics
import matplotlib.pyplot as plt
import datetime

exponential_creation_days = 9
exponential_delivery_days = 5
exponential_delivery_confirmation_days = 7
dataset = SyntheticDataGen(
    p_fraud_seller = .2, 
    beta_lower_cs = 2, 
    beta_higher_cs = 10, 
    exponential_creation = 60*60*24*exponential_creation_days, 
    exponential_delivery = 60*60*24*exponential_delivery_days, 
    exponential_delivery_confirmation=60*60*24*exponential_delivery_confirmation_days
)

dataset.cutoff_day = 1000000

num_orders = []
deltatime_creation = []
deltatime_delivery = []
deltatime_delivery_confirmation = []
classification_score = []
for _ in range(100000):
    seller = dataset.sample_seller()
    classification_score.append(seller.classifier_score)
    num_orders.append(len(seller.orders))
    deltatime_creation_signup = []
    for order in seller.orders:
        deltatime_creation_signup.append((order.creation_timestamp.date() - seller.signup_timestamp.date()))
        if order.delivery_date is not None:
            #delivery_datetime = datetime.datetime.combine(order.delivery_date, datetime.time(0,0,0))
            #deltatime_delivery.append((delivery_datetime - order.creation_timestamp).days)
            deltatime_delivery.append((order.delivery_date - order.creation_timestamp.date()).days)
        if order.delivery_confirmation_timestamp is not None:
            #deltatime_delivery_confirmation.append((order.delivery_confirmation_timestamp - delivery_datetime).days)
            deltatime_delivery_confirmation.append((order.delivery_confirmation_timestamp.date() - order.delivery_date).days)
    deltatime_creation_signup.sort()
    deltatime_creation.append((deltatime_creation_signup[0]).days)
    for idx in range(1, len(deltatime_creation_signup)):
        deltatime_creation.append((deltatime_creation_signup[idx] - deltatime_creation_signup[idx-1]).days)

print(f'mean # orders: {statistics.mean(num_orders)}')
#
#plt.hist(classification_score,bins=50)
#plt.show()
#NOTE: classification score is still very predictive, 
# - it might make sense to have blindspots (i.e. stuff with cs > .8 is not observed in training)
# - it might make sense to have higher uncertanty to avoid easy suboptimal policies
#
#plt.hist(deltatime_creation,bins=max(deltatime_creation))
#plt.show()
print(f'mean delta_creation: exp={exponential_creation_days} vs real={statistics.mean(deltatime_creation)}')
#
#plt.hist(deltatime_delivery,bins=max(deltatime_delivery))
#plt.show()
print(f'mean delta_creation: exp={exponential_delivery_days} vs real={statistics.mean(deltatime_delivery)}')
#
plt.hist(deltatime_delivery_confirmation,bins=max(deltatime_delivery_confirmation))
plt.show()
print(f'mean delta_creation: exp={exponential_delivery_confirmation_days} vs real={statistics.mean(deltatime_delivery_confirmation)}')

# testing dataframes
sellers_df, discovery_df, orders_df = dataset.get_dataset(100)
breakpoint()