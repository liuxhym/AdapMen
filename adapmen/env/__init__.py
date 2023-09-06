from __future__ import annotations
#from tty import setraw

from typing import Optional

import gym
import numpy as np
import yaml

from adapmen.env.absorbing_wrapper import AbsorbingWrapper
from adapmen.env.human_in_the_loop_env import HumanInTheLoopEnv
from adapmen.env.normalize_action_wrapper import check_and_normalize_box_actions
from adapmen.env.normalize_action_wrapper import NormalizeBoxActionWrapper
from adapmen.env.normalize_state_wrapper import NormalizeStateWrapper
from adapmen.env.expert_guided_env import ExpertGuidedEnv
from unstable_baselines.common.env_wrapper import wrap_atari_env
from gym import register
from metadrive.engine.engine_utils import get_global_config
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.engine.core.manual_controller import KeyboardController, SteeringWheelController

MUJOCO_ENVS =['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3', 'Ant-v3',
    'CarPush-v0',
    'CarGoal-v0',
    'DoggoPush-v0',
    'DoggoGoal-v0']

ATARI_ENVS = ['MsPacman-v4',   'BeamRider-v4',  "DemonAttack-v4", "Pong-v4", "Qbert-v4", "Enduro-v4"]
METADRIVE_ENVS = ['metadrive', 'HumanInTheLoopMetaDrive', 'ExpertInTheLoopMetaDrive', 'HumanDAggerMetaDrive']
DISCRETE_ENVS = ['CartPole-v1']

EXPERT_PERFORMANCE ={
    'MsPacman-v4': 1626,
    'Seaquest-v4': 504, 
    'BeamRider-v4': 1472,
    "TimePilot-v4": 133,
    "SpaceInvaders-v4":222,
    "DemonAttack-v4":343,
    "Pong-v4": 21,
    "Qbert-v4":1250,
    "Enduro-v4":203,
    "Boxing-v4":3,
    "Freeway-v4":22
}
class DAggerPolicy(EnvInputPolicy):
    """
    Record the takeover signal
    """
    def __init__(self, obj, seed):
        super(DAggerPolicy, self).__init__(obj, seed)
        config = get_global_config()
        if config["manual_control"] and config["use_render"]:
            if config["controller"] == "joystick":
                self.controller = SteeringWheelController()
            elif config["controller"] == "keyboard":
                self.controller = KeyboardController(False)
            else:
                raise ValueError("Unknown Policy: {}".format(config["controller"]))
        self.takeover = False

    def act(self, agent_id):
        agent_action = super(DAggerPolicy, self).act(agent_id)
        if self.engine.global_config["manual_control"] and self.engine.agent_manager.get_agent(
                agent_id) is self.engine.current_track_vehicle and not self.engine.main_camera.is_bird_view_camera():
            expert_action = self.controller.process_input(self.engine.current_track_vehicle)
            if isinstance(self.controller, SteeringWheelController) and (self.controller.left_shift_paddle
                                                                         or self.controller.right_shift_paddle):
                # if expert_action[0]*agent_action[0]< 0 or expert_action[1]*agent_action[1] < 0:
                self.takeover = True
                return expert_action
            elif isinstance(self.controller, KeyboardController) and abs(sum(expert_action)) > 1e-2:
                self.takeover = True
        self.takeover = False
        return agent_action, expert_action
    
    
metadrive_sac_expert_config = dict(
        vehicle_config=dict(
            use_saver=False,
            free_level=100),
        safe_rl_env=True,
        controller="keyboard",
        use_render=False,
        manual_control=False,
        #traffic_density=0.1,
        random_traffic=False,
        environment_num=100,
        random_agent_model=False,
        random_lane_width=False,
        random_lane_num=False,
        map=4,  # seven block
        start_seed=0,
    )

metadrive_human_expert_config = dict(
        use_render=True, 
        main_exp=True, 
        horizon=1500, 
        safe_rl_env=True,
        controller="keyboard",
        manual_control=True,
        #traffic_density=0.1,
        random_traffic=False,
        environment_num=100,
        random_agent_model=False,
        random_lane_width=False,
        random_lane_num=False,
        map=4,  # seven block
        start_seed=0,
)

human_dagger_config = dict(
        controller="keyboard",
        use_render=True,
        main_exp=True, 
        manual_control=True,
        safe_rl_env=True,
        horizon=1500,
        #traffic_density=0.1,
        environment_num=100,
        random_traffic=False,
        random_agent_model=False,
        random_lane_width=False,
        random_lane_num=False,
        map=4,  # seven block
        start_seed=0,
        agent_policy=DAggerPolicy
)

def create_il_env(env_name: str, seed: Optional[int], mean: Optional[np.ndarray], std: Optional[np.ndarray], add_absorbing_state=False, **kwargs):
    if env_name in MUJOCO_ENVS:
       
        env = gym.make(env_name)
        env = check_and_normalize_box_actions(env)
        env.seed(seed)

        if mean is not None and std is not None:
            shift = -mean
            scale = 1.0 / std
            env = NormalizeStateWrapper(env, shift=shift, scale=scale)

        setattr(env, 'discrete', False)
        if add_absorbing_state:
            return AbsorbingWrapper(env)
        else:
            return env
    elif env_name in METADRIVE_ENVS:
        if env_name == "ExpertInTheLoopMetaDrive" or env_name == 'metadrive':
            env = ExpertGuidedEnv(metadrive_sac_expert_config)
        elif env_name == "HumanInTheLoopMetaDrive":
            env = HumanInTheLoopEnv(metadrive_human_expert_config)
        elif env_name == "HumanDAggerMetaDrive":
            env = HumanInTheLoopEnv(human_dagger_config)
        setattr(env, 'discrete', False)
        return env
    elif env_name in ATARI_ENVS:
        env = gym.make(env_name)
        #env = gym.make(env_name, render_mode="human")
        #env.seed(seed)

        if mean is not None and std is not None:
            shift = -mean
            scale = 1.0 / std
            env = NormalizeStateWrapper(env, shift=shift, scale=scale)
        env = wrap_atari_env(env, (84, 84), 30, 4, 4)
        setattr(env, 'discrete', True)
        return env
    elif env_name in DISCRETE_ENVS:
        env = gym.make(env_name)
        #env = gym.make(env_name, render_mode="human")
        env.seed(seed)

        if mean is not None and std is not None:
            shift = -mean
            scale = 1.0 / std
            env = NormalizeStateWrapper(env, shift=shift, scale=scale)
        setattr(env, 'discrete', True)
        return env
        
    else:
        raise NotImplementedError


