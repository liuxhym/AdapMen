default_args ={
  "task_name": 'bc',
  "algo": 'bc',
  "log_interval": 100,
  "eval_interval": 500,
  "snapshot_interval": 10000,

  "sac":{
    "actor_network_params": [("mlp", 256), ("mlp", 256)]
  },
  "env":{
    "env_name" : '',
    "discount" : 0.99,
    "num_traj": 50,
    "expert_data_path": '',
    "use_normalize_states": False,
    "subsample_rate": 1.0,
  },
  "bc":{
    "actor_lr" : 3e-4,
    "batch_size": 32,
    "start_training_timesteps": 0,
    "max_timesteps": 1000000,
  }
}