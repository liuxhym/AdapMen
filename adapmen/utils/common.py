import json
import math
import gym
import numpy as np
from metadrive.obs.observation_base import ObservationBase
from metadrive.utils import clip


class StateObservation(ObservationBase):
    def __init__(self, config):
        super(StateObservation, self).__init__(config)

    @property
    def observation_space(self):
        # Navi info + Other states
        shape = 19
        return gym.spaces.Box(-0.0, 1.0, shape=(shape,), dtype=np.float32)

    def observe(self, vehicle):
        navi_info = vehicle.navigation.get_navi_info()
        ego_state = self.vehicle_state(vehicle)
        ret = np.concatenate([ego_state, navi_info])
        return ret.astype(np.float32)

    def vehicle_state(self, vehicle):
        # update out of road
        current_reference_lane = vehicle.navigation.current_ref_lanes[-1]
        lateral_to_left, lateral_to_right = vehicle.dist_to_left_side, vehicle.dist_to_right_side
        total_width = float(
            (vehicle.navigation.map.config["lane_num"] + 1) * vehicle.navigation.map.config["lane_width"]
        )
        info = [
            clip(lateral_to_left / total_width, 0.0, 1.0),
            clip(lateral_to_right / total_width, 0.0, 1.0),
            vehicle.heading_diff(current_reference_lane),
            # Note: speed can be negative denoting free fall. This happen when emergency brake.
            clip((vehicle.speed + 1) / (vehicle.max_speed + 1), 0.0, 1.0),
            clip((vehicle.steering / vehicle.max_steering + 1) / 2, 0.0, 1.0),
            clip((vehicle.last_current_action[0][0] + 1) / 2, 0.0, 1.0),
            clip((vehicle.last_current_action[0][1] + 1) / 2, 0.0, 1.0)
        ]
        heading_dir_last = vehicle.last_heading_dir
        heading_dir_now = vehicle.heading
        cos_beta = heading_dir_now.dot(heading_dir_last
                                       ) / (np.linalg.norm(heading_dir_now) * np.linalg.norm(heading_dir_last))

        beta_diff = np.arccos(clip(cos_beta, 0.0, 1.0))

        # print(beta)
        yaw_rate = beta_diff / 0.1
        # print(yaw_rate)
        info.append(clip(yaw_rate, 0.0, 1.0))
        _, lateral = vehicle.lane.local_coordinates(vehicle.position)
        info.append(clip((lateral * 2 / vehicle.navigation.map.config["lane_width"] + 1.0) / 2.0, 0.0, 1.0))
        return info


class ExpertObservation(ObservationBase):
    def __init__(self, vehicle_config):
        self.state_obs = StateObservation(vehicle_config)
        super(ExpertObservation, self).__init__(vehicle_config)
        self.cloud_points = None
        self.detected_objects = None

    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        if self.config["lidar"]["num_lasers"] > 0 and self.config["lidar"]["distance"] > 0:
            # Number of lidar rays and distance should be positive!
            shape[0] += self.config["lidar"]["num_lasers"] + self.config["lidar"]["num_others"] * 4
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def observe(self, vehicle):
        state = self.state_observe(vehicle)
        other_v_info = self.lidar_observe(vehicle)
        self.current_observation = np.concatenate((state, np.asarray(other_v_info)))
        ret = self.current_observation
        return ret.astype(np.float32)

    def state_observe(self, vehicle):
        return self.state_obs.observe(vehicle)

    def lidar_observe(self, vehicle):
        other_v_info = []
        if vehicle.lidar.available:
            cloud_points, detected_objects = vehicle.lidar.perceive(vehicle, )
            other_v_info += vehicle.lidar.get_surrounding_vehicles_info(
                vehicle, detected_objects, 4)
            other_v_info += cloud_points
            self.cloud_points = cloud_points
            self.detected_objects = detected_objects
        return other_v_info

def normpdf(x, mean, sd):
    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


def expert_action_prob(action, obs, weights, deterministic=False):
    obs = obs.reshape(1, -1)
    x = np.matmul(obs, weights["default_policy/fc_1/kernel"]) + weights["default_policy/fc_1/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/fc_2/kernel"]) + weights["default_policy/fc_2/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/fc_out/kernel"]) + weights["default_policy/fc_out/bias"]
    x = x.reshape(-1)
    mean, log_std = np.split(x, 2)
    std = np.exp(log_std)
    a_0_p = normpdf(action[0], mean[0], std[0])
    a_1_p = normpdf(action[1], mean[1], std[1])
    expert_action = np.random.normal(mean, std) if not deterministic else mean
    return expert_action, a_0_p, a_1_p


def load_weights(path: str):
    """
    Load NN weights
    :param path: weights file path path
    :return: NN weights object
    """
    # try:
    model = np.load(path)
    return model
    # except FileNotFoundError:
    # print("Can not find {}, didn't load anything".format(path))
    # return None
