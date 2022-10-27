import datetime
from environment import Order, FakeSellerGymEnv, Action, _get_state_of_orders_at_timestep, AVERAGE_COST_PER_ORDER, AVERAGE_BENEFIT_PER_ORDER, COST_INVESTIGATION
import numpy as np 
import pytest
import pandas as pd

ts = datetime.datetime.fromisoformat
date = datetime.date.fromisoformat

def test__get_state_of_orders_at_timestep_with_expected_inputs(signup_timestamp):
    orders = [
        Order(ts("2023-01-03 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
    ]
    creation, delivery, delivery_confirmation = _get_state_of_orders_at_timestep(signup_timestamp, orders, timestep=5)

    expected_creation = np.zeros(7)
    expected_creation[5] = 1
    expected_delivery = np.zeros(14)
    expected_delivery[8] = 1
    expected_delivery_confirmation = np.zeros(7)

    np.testing.assert_allclose(creation, expected_creation)
    np.testing.assert_allclose(delivery, expected_delivery)
    np.testing.assert_allclose(delivery_confirmation, expected_delivery_confirmation)


def test__get_state_of_orders_at_timestep_with_empty_orders(signup_timestamp):
    orders = []
    creation, delivery, delivery_confirmation = _get_state_of_orders_at_timestep(signup_timestamp, orders, timestep=5)

    expected_creation = np.zeros(7)
    expected_delivery = np.zeros(14)
    expected_delivery_confirmation = np.zeros(7)
    np.testing.assert_allclose(creation, expected_creation)
    np.testing.assert_allclose(delivery, expected_delivery)
    np.testing.assert_allclose(delivery_confirmation, expected_delivery_confirmation)



def test__get_state_of_orders_at_timestep_with_expected_inputs_timestep_after_confirmation(signup_timestamp):
    orders = [
        Order(ts("2023-01-03 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
    ]
    creation, delivery, delivery_confirmation = _get_state_of_orders_at_timestep(signup_timestamp, orders, timestep=8)

    expected_creation = np.zeros(7)
    expected_creation[3] = 1
    expected_delivery = np.zeros(14)
    expected_delivery[5] = 1
    expected_delivery_confirmation = np.zeros(7)
    expected_delivery_confirmation[6] = 1

    np.testing.assert_allclose(creation, expected_creation)
    np.testing.assert_allclose(delivery, expected_delivery)
    np.testing.assert_allclose(delivery_confirmation, expected_delivery_confirmation)


def test__get_state_of_orders_at_timestep_with_expected_inputs_times_2(signup_timestamp):
    orders = [
        Order(ts("2023-01-03 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
    ]*2
    creation, delivery, delivery_confirmation = _get_state_of_orders_at_timestep(signup_timestamp, orders, timestep=5)

    expected_creation = np.zeros(7)
    expected_creation[5] = 2
    expected_delivery = np.zeros(14)
    expected_delivery[8] = 2
    expected_delivery_confirmation = np.zeros(7)

    np.testing.assert_allclose(creation, expected_creation)
    np.testing.assert_allclose(delivery, expected_delivery)
    np.testing.assert_allclose(delivery_confirmation, expected_delivery_confirmation)


def test__get_state_of_orders_at_timestep_returns_all_zeros_when_no_orders_at_current_timestep(signup_timestamp):
    orders = [
        Order(ts("2023-01-03 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
    ]
    creation, delivery, delivery_confirmation = _get_state_of_orders_at_timestep(signup_timestamp, orders, timestep=0)

    expected_creation = np.zeros(7)
    expected_delivery = np.zeros(14)
    expected_delivery_confirmation = np.zeros(7)
    np.testing.assert_allclose(creation, expected_creation)
    np.testing.assert_allclose(delivery, expected_delivery)
    np.testing.assert_allclose(delivery_confirmation, expected_delivery_confirmation)


def test__get_state_of_orders_at_timestep_when_order_at_same_day_as_signup_for_timestep_0(signup_timestamp):
    orders = [
        Order(ts("2023-01-01 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
    ]
    creation, delivery, delivery_confirmation = _get_state_of_orders_at_timestep(signup_timestamp, orders, timestep=0)

    expected_creation = np.zeros(7)
    expected_delivery = np.zeros(14)
    expected_delivery_confirmation = np.zeros(7)
    np.testing.assert_allclose(creation, expected_creation)
    np.testing.assert_allclose(delivery, expected_delivery)
    np.testing.assert_allclose(delivery_confirmation, expected_delivery_confirmation)


def test__get_state_of_orders_at_timestep_when_order_at_same_day_as_signup_for_timestep_1(signup_timestamp):
    orders = [
        Order(ts("2023-01-01 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
    ]
    creation, delivery, delivery_confirmation = _get_state_of_orders_at_timestep(signup_timestamp, orders, timestep=1)

    expected_creation = np.zeros(7)
    expected_creation[6] = 1 
    expected_delivery = np.zeros(14)
    expected_delivery[10] = 1
    expected_delivery_confirmation = np.zeros(7)
    np.testing.assert_allclose(creation, expected_creation)
    np.testing.assert_allclose(delivery, expected_delivery)
    np.testing.assert_allclose(delivery_confirmation, expected_delivery_confirmation)


@pytest.mark.parametrize("timestep, expected_last_bin", 
    [
        (7, 0), 
        (8, 1)
    ]
)
def test__get_state_of_orders_at_timestep_when_for_delivery_during_timestep_7(signup_timestamp, timestep, expected_last_bin):
    orders = [
        Order(ts("2023-01-03 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
    ]
    _, _, delivery_confirmation = _get_state_of_orders_at_timestep(signup_timestamp, orders, timestep=timestep)
    expected_delivery_confirmation = np.array([0, 0, 0, 0, 0, 0, expected_last_bin])

    np.testing.assert_allclose(delivery_confirmation, expected_delivery_confirmation)


def test_fake_seller_gym_env__reset():
    signup_timestamp = ts("2023-01-01 13:12:00")
    classifier_score =  0.1
    discovery_timestamp = ts("2023-01-08 16:00:00")
    sellers_df  = pd.DataFrame(
        columns=["seller_id", "signup_timestamp", "classifier_score"],
        data=[
            (101, signup_timestamp, classifier_score)
        ]
    )
    discorvery_df  = pd.DataFrame(
        columns=["seller_id", "discovery_timestamp"],
        data=[
            (101, discovery_timestamp)
        ]
    )
    order = Order(ts("2023-01-03 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
    orders_df  = pd.DataFrame(
        columns=["seller_id", "creation_timestamp", "delivery_date", "delivery_confirmation_timestamp"],
        data=[
            (101, order.creation_timestamp, order.delivery_date, order.delivery_confirmation_timestamp)
        ]
    )
    gym_env = FakeSellerGymEnv(sellers_df, discorvery_df, orders_df)

    obs, info = gym_env.reset()
    expected_obs = np.zeros(30)
    expected_obs[29] = classifier_score

    np.testing.assert_allclose(obs, expected_obs)
    assert info == {}
    assert gym_env.signup_timestamp == signup_timestamp
    assert gym_env.classifier_score == classifier_score
    assert gym_env.orders == [order]
    assert gym_env.discovery_timestamp == discovery_timestamp
    assert gym_env.final_timestep == 8
    

def test_fake_seller_gym_step_done_when_investigate(gym_for_fake_seller):
    _, _, done, _ = gym_for_fake_seller.step(action=Action.INVESTIGATE)
    assert done


def test_fake_seller_gym__reward_for_fake_seller__when_always_allow(gym_for_fake_seller):
    done = False
    total_reward = 0
    while not done:
        _, reward, done, _ = gym_for_fake_seller.step(Action.ALLOW)
        total_reward += reward
    assert total_reward ==  -AVERAGE_COST_PER_ORDER

@pytest.mark.parametrize("investigation_timestep, expected_return", [
    (1, -COST_INVESTIGATION),
    (2, -COST_INVESTIGATION),
    (3, -COST_INVESTIGATION-AVERAGE_COST_PER_ORDER),
    (4, -COST_INVESTIGATION-AVERAGE_COST_PER_ORDER),
    (5, -COST_INVESTIGATION-AVERAGE_COST_PER_ORDER),
    (6, -COST_INVESTIGATION-AVERAGE_COST_PER_ORDER),
    (7, -COST_INVESTIGATION-AVERAGE_COST_PER_ORDER),
    (8, -AVERAGE_COST_PER_ORDER),
])
def test_fake_seller_gym__reward_for_fake_seller__when_investigate_once(gym_for_fake_seller, investigation_timestep, expected_return):
    done = False
    total_reward = 0
    while not done:
        action = Action.INVESTIGATE if gym_for_fake_seller.timestep == investigation_timestep else Action.ALLOW
        _, reward, done, _ = gym_for_fake_seller.step(action)
        total_reward += reward
    assert total_reward == expected_return


def test_fake_seller_gym__reward_for_genuine_seller__when_always_allow(gym_for_genuine_seller):
    done = False
    rewards = [0]*91
    while not done:
        _, reward, done, _ = gym_for_genuine_seller.step(action=Action.ALLOW)
        rewards[gym_for_genuine_seller.timestep] = reward
    expected_reward = [0]*91
    expected_reward[8] = AVERAGE_BENEFIT_PER_ORDER
    assert rewards == expected_reward

def test_fake_seller_gym__reward_for_genuine_seller__when_always_allow_v2(gym_for_genuine_seller):
    done = False
    rewards = []
    while not done:
        _, reward, done, _ = gym_for_genuine_seller.step(action=Action.ALLOW)
        rewards.append(reward)
    expected_reward = [0]*90
    expected_reward[7] = AVERAGE_BENEFIT_PER_ORDER
    assert rewards == expected_reward


@pytest.fixture
def signup_timestamp():
    return ts("2023-01-01 10:00:00")


@pytest.fixture
def gym_for_fake_seller(signup_timestamp):
    sellers_df  = pd.DataFrame(
        columns=["seller_id", "signup_timestamp", "classifier_score"],
        data=[
            (101, signup_timestamp, 0.9)
        ]
    )
    discorvery_df  = pd.DataFrame(
        columns=["seller_id", "discovery_timestamp"],
        data=[
            (101, ts("2023-01-08 16:00:00"))
        ]
    )
    orders_df  = pd.DataFrame(
        columns=["seller_id", "creation_timestamp", "delivery_date", "delivery_confirmation_timestamp"],
        data=[
            (101, ts("2023-01-03 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
        ]
    )
    gym_for_fake_seller = FakeSellerGymEnv(sellers_df, discorvery_df, orders_df)
    gym_for_fake_seller.reset()
    return gym_for_fake_seller


@pytest.fixture
def gym_for_genuine_seller(signup_timestamp):
    sellers_df  = pd.DataFrame(
        columns=["seller_id", "signup_timestamp", "classifier_score"],
        data=[
            (101, signup_timestamp, 0.1)
        ]
    )
    discorvery_df  = pd.DataFrame(
        columns=["seller_id", "discovery_timestamp"],
        data=[
        ]
    )
    orders_df  = pd.DataFrame(
        columns=["seller_id", "creation_timestamp", "delivery_date", "delivery_confirmation_timestamp"],
        data=[
            (101, ts("2023-01-03 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
        ]
    )
    gym_for_fake_seller = FakeSellerGymEnv(sellers_df, discorvery_df, orders_df)
    gym_for_fake_seller.reset()
    return gym_for_fake_seller