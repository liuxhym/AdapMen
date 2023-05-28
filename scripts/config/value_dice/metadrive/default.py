default_args ={
  "task_name": 'value_dice_cp',
  "algo": 'value_dice',
  "log_interval": 500,
  "eval_interval": 500,
  "snapshot_interval": 5000000,
  "buffer_size": 50000,
  "teacher": "dqn",
  "horizon": 100,

  "sac":{
    "actor_network_params": [("mlp", 256), ("mlp", 256)]
  },
  "env":{
    "env_name" : '',
    "discount" : 0.99,
    "num_traj": 100,
    "expert_data_path": 'data/Metadrive_Sac_100.npz',
    "use_normalize_states": False,
    "subsample_rate": 1.0,
  },
  "value_dice":{
    "nu_network_params": [("mlp", 256), ("mlp", 256)],
    "actor_lr": 1e-5,
    "nu_lr": 1e-3,
    "replay_reg_coeff": 0.1,
    "nu_reg_coeff": 10.0,
    "start_training_timesteps" : 2000,
    "batch_size": 32,
    "max_timesteps": 30000,
    "num_random_actions": 2000,
    "absorbing_per_episode": 10,
    "updates_interval": 200, 
    "updates_per_step": 50
  }
}