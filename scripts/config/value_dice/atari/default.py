default_args ={
  "task_name": 'value_dice_cp',
  "algo": 'value_dice',
  "log_interval": 1000,
  "eval_interval": 2000,
  "snapshot_interval": 5000000,
  "buffer_size": 50000,
  "teacher": "dqn",
  "horizon": 100,

  "sac":{
    "actor_network_params": [("conv2d", 16, 8, 4, 0), ("conv2d", 32, 4, 2, 0),("flatten",), ("mlp", 256), ("mlp", 256)]
  },
  "env":{
    "env_name" : '',
    "discount" : 0.99,
    "num_traj": 100,
    "expert_data_path": '',
    "use_normalize_states": False,
    "subsample_rate": 1.0,
  },
  "value_dice":{
    "nu_network_params": [("conv2d", 16, 8, 4, 0), ("conv2d", 32, 4, 2, 0),("flatten",), ("mlp", 256), ("mlp", 128)],
    "actor_lr": 1e-5,
    "nu_lr": 1e-3,
    "replay_reg_coeff": 0.1,
    "nu_reg_coeff": 10.0,
    "start_training_timesteps" : 1000,
    "batch_size": 32,
    "max_timesteps": 500000,
    "num_random_actions": 2000,
    "absorbing_per_episode": 10,
    "updates_per_step": 5
  }
}