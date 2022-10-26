import datetime
from environment import Order, FakeSellerGymEnv, Action
import numpy as np 
import pytest
import pandas as pd

ts = datetime.datetime.fromisoformat
date = datetime.date.fromisoformat

def test_environment___get_state_of_orders_at_timestep_with_expected_inputs(fake_seller_gym):
    orders = [
        Order(ts("2023-01-03 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
    ]

    fake_seller_gym.orders = orders


    creation, delivery, delivery_confirmation = fake_seller_gym._get_state_of_orders_at_timestep(timestep=5)
    expected_creation = np.zeros(8)
    expected_creation[5] = 1
    expected_delivery = np.zeros(14)
    expected_delivery[8] = 1
    expected_delivery_confirmation = np.zeros(8)
    np.testing.assert_allclose(creation, expected_creation)
    np.testing.assert_allclose(delivery, expected_delivery)
    np.testing.assert_allclose(delivery_confirmation, expected_delivery_confirmation)


def test_environment___get_state_of_orders_at_timestep_with_empty_orders(fake_seller_gym):
    orders = []

    fake_seller_gym.orders = orders
    creation, delivery, delivery_confirmation = fake_seller_gym._get_state_of_orders_at_timestep(timestep=5)
    expected_creation = np.zeros(8)
    expected_delivery = np.zeros(14)
    expected_delivery_confirmation = np.zeros(8)
    np.testing.assert_allclose(creation, expected_creation)
    np.testing.assert_allclose(delivery, expected_delivery)
    np.testing.assert_allclose(delivery_confirmation, expected_delivery_confirmation)



def test_environment___get_state_of_orders_at_timestep_with_expected_inputs_timestep_after_confirmation(fake_seller_gym):
    orders = [
        Order(ts("2023-01-03 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
    ]

    fake_seller_gym.orders = orders


    creation, delivery, delivery_confirmation = fake_seller_gym._get_state_of_orders_at_timestep(timestep=8)
    expected_creation = np.zeros(8)
    expected_creation[3] = 1
    expected_delivery = np.zeros(14)
    expected_delivery[5] = 1
    expected_delivery_confirmation = np.zeros(8)
    expected_delivery_confirmation[6] = 1
    np.testing.assert_allclose(creation, expected_creation)
    np.testing.assert_allclose(delivery, expected_delivery)
    np.testing.assert_allclose(delivery_confirmation, expected_delivery_confirmation)


def test_environment___get_state_of_orders_at_timestep_with_expected_inputs_times_2(fake_seller_gym):
    orders = [
        Order(ts("2023-01-03 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
    ]*2

    fake_seller_gym.orders = orders


    creation, delivery, delivery_confirmation = fake_seller_gym._get_state_of_orders_at_timestep(timestep=5)
    expected_creation = np.zeros(8)
    expected_creation[5] = 2
    expected_delivery = np.zeros(14)
    expected_delivery[8] = 2
    expected_delivery_confirmation = np.zeros(8)
    np.testing.assert_allclose(creation, expected_creation)
    np.testing.assert_allclose(delivery, expected_delivery)
    np.testing.assert_allclose(delivery_confirmation, expected_delivery_confirmation)


def test_environment___get_state_of_orders_at_timestep_returns_all_zeros_when_no_orders_at_current_timestep(fake_seller_gym):
    orders = [
        Order(ts("2023-01-03 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
    ]

    fake_seller_gym.orders = orders


    creation, delivery, delivery_confirmation = fake_seller_gym._get_state_of_orders_at_timestep(timestep=0)
    expected_creation = np.zeros(8)
    expected_delivery = np.zeros(14)
    expected_delivery_confirmation = np.zeros(8)
    np.testing.assert_allclose(creation, expected_creation)
    np.testing.assert_allclose(delivery, expected_delivery)
    np.testing.assert_allclose(delivery_confirmation, expected_delivery_confirmation)


def test_environment___get_state_of_orders_at_timestep_when_order_at_same_day_as_signup_for_timestep_0(fake_seller_gym):
    orders = [
        Order(ts("2023-01-01 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
    ]

    fake_seller_gym.orders = orders


    creation, delivery, delivery_confirmation = fake_seller_gym._get_state_of_orders_at_timestep(timestep=0)
    expected_creation = np.zeros(8)
    expected_delivery = np.zeros(14)
    expected_delivery_confirmation = np.zeros(8)
    np.testing.assert_allclose(creation, expected_creation)
    np.testing.assert_allclose(delivery, expected_delivery)
    np.testing.assert_allclose(delivery_confirmation, expected_delivery_confirmation)


def test_environment___get_state_of_orders_at_timestep_when_order_at_same_day_as_signup_for_timestep_1(fake_seller_gym):
    orders = [
        Order(ts("2023-01-01 12:00:00"), date("2023-01-07"), ts("2023-01-08 11:00:00"))
    ]

    fake_seller_gym.orders = orders


    creation, delivery, delivery_confirmation = fake_seller_gym._get_state_of_orders_at_timestep(timestep=1)
    expected_creation = np.zeros(8)
    expected_creation[6] = 1 
    expected_delivery = np.zeros(14)
    expected_delivery[10] = 1
    expected_delivery_confirmation = np.zeros(8)
    np.testing.assert_allclose(creation, expected_creation)
    np.testing.assert_allclose(delivery, expected_delivery)
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
    expected_obs = np.zeros(32)
    expected_obs[31] = classifier_score

    np.testing.assert_allclose(obs, expected_obs)
    assert info == {}
    assert gym_env.signup_timestamp == signup_timestamp
    assert gym_env.classifier_score == classifier_score
    assert gym_env.orders == [order]
    assert gym_env.discovery_timestamp == discovery_timestamp
    
def test_fake_seller_gym_step_done_when_investigate(fake_seller_gym):
    _, _, done, _ = fake_seller_gym.step(action=Action.INVESTIGATE)
    assert done
    

@pytest.fixture
def fake_seller_gym():
    fake_seller_gym = FakeSellerGymEnv(None, None, None)
    fake_seller_gym.discovery_timestamp = None
    fake_seller_gym.signup_timestamp = ts("2023-01-01 10:00:00")
    return fake_seller_gym


@pytest.fixture
def gym_for_fake_seller():
    sellers_df  = pd.DataFrame(
        columns=["seller_id", "signup_timestamp", "classifier_score"],
        data=[
            (101, ts("2023-01-01 13:12:00"), 0.9)
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
    gym_env = FakeSellerGymEnv(sellers_df, discorvery_df, orders_df)
    gym_env.reset()
    return gym_env