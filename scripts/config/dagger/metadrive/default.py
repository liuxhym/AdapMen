default_args ={
  "task_name": 'DAgger',
  "algo": 'DAgger',
  "log_interval": 500,
  "eval_interval": 500,
  "snapshot_interval": 500,
  "buffer_size": 2000,

  "sac":{
    "actor_update_interval": 1,
    "actor_network_params": [("mlp", 256), ("mlp", 256)]
  },
  "env":{
    "env_name" : '',
    "discount" : 0.99,
    "num_traj": 0,
    "expert_data_path": '',
    "use_normalize_states": False
  },
  "dagger":{
    "actor_lr" : 1e-4,
    "batch_size": 128,
    "bc_step": 0,
    "max_timesteps": 30000,
    "updates_per_step": 50,
    "samplesteps": 2000,
    "update_interval": 200
  }
}