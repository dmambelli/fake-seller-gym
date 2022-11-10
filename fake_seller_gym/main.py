from simulator.environment import FakeSellerGymEnv, Action, PAST_BINS
from synthetic_data_gen.generator import SyntheticDataGen
import numpy as np
import statistics

def policy_wait_expected_delivery_conf_delay_to_investigate(obs, expected_delivery_conf_delay):
    # oracle policy with access to aggregated statics of fraud
    
    # break down observation vector
    creations = obs[:7]
    deliveries = obs[7:21]
    delivery_confirmations = obs[21:28]
    # check if there is any delivery or confirmation delivery
    expected_delivery_conf_delay_bin = np.nonzero(np.histogram(-expected_delivery_conf_delay, bins=PAST_BINS)[0])[0][0]
    delivery_too_old = np.sum(deliveries[:expected_delivery_conf_delay_bin]) > 0
    delivery_confirmation_exists = np.sum(delivery_confirmations) > 0
    # picking action
    if delivery_too_old and not delivery_confirmation_exists:
        action = Action.INVESTIGATE
    else:
        action = Action.ALLOW
        
    return action

def policy_look_at_classification_score(obs, classification_score_threshold):
    # break down observation vector
    classification_score = obs[-1]
    # picking action
    if classification_score > classification_score_threshold:
        action = Action.INVESTIGATE
    else:
        action = Action.ALLOW
        
    return action

def policy_optimal(obs, fraud):
    # oracle policy
    if fraud:
        action = Action.INVESTIGATE
    else:
        action = Action.ALLOW
        
    return action

exponential_creation_days = 5
exponential_delivery_days = 7
exponential_delivery_confirmation_days = 6
dataset = SyntheticDataGen(
    p_fraud_seller = .2, 
    beta_lower_cs = 2, 
    beta_higher_cs = 10, 
    exponential_creation = 60*60*24*exponential_creation_days, 
    exponential_delivery = 60*60*24*exponential_delivery_days, 
    exponential_delivery_confirmation=60*60*24*exponential_delivery_confirmation_days
)

sellers_df, discovery_df, orders_df = dataset.get_dataset(1000)
gym_for_fake_seller = FakeSellerGymEnv(sellers_df, discovery_df, orders_df)

num_resets = 1000

mean_reward = []
mean_traj_length_fraud = []

for episode in range(num_resets):
    obs, info = gym_for_fake_seller.reset()
    done = False
    tot_reward = 0
    while not done:
        #action = policy_wait_expected_delivery_conf_delay_to_investigate(obs, exponential_delivery_confirmation_days+20)
        action = policy_optimal(obs, gym_for_fake_seller.discovery_timestamp is not None)
        obs, reward, done, info = gym_for_fake_seller.step(action)
        tot_reward += reward
        #print(f't{gym_for_fake_seller.timestep}: {action.name}')
            
    # priviledged infos
    fraud = gym_for_fake_seller.discovery_timestamp is not None
    print(f'episode {episode} -> tot_rew={tot_reward} in {gym_for_fake_seller.timestep} timesteps [fraud:{fraud}]')
    mean_reward.append(tot_reward)
    if gym_for_fake_seller.discovery_timestamp is not None:
        mean_traj_length_fraud.append(gym_for_fake_seller.timestep)
    
print(f'average return: {statistics.mean(mean_reward)}')
print(f'fraudster caught in {statistics.mean(mean_traj_length_fraud)} timesteps')