import datetime
from environment import Order, FakeSellerGymEnv
import numpy as np 
import pytest


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


@pytest.fixture
def fake_seller_gym():
    fake_seller_gym = FakeSellerGymEnv(None, None, None)
    fake_seller_gym.discovery_timestamp = None
    fake_seller_gym.signup_timestamp = ts("2023-01-01 10:00:00")
    return fake_seller_gym