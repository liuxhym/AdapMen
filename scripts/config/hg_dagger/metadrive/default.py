default_args ={
  "task_name": 'DAgger',
  "algo": 'DAgger',
  "log_interval": 500,
  "eval_interval": 500000000000000,
  "snapshot_interval": 500,
  "buffer_size": 2000,
  "variance_threshold": 0.01,
  "discrepacy_threshold": 0.8,

  "actor":{
    "num_actors": 5,
    "actor_network_params": [("mlp", 256), ("mlp", 256)]
  },
  "env":{
    "env_name": '',
    "discount": 0.99,
    "num_traj": 0,
    "use_normalize_states": False,
    "expert_data_path": ''
  },
  "dagger":{
    "actor_lr": 1e-4,
    "batch_size": 128,
    "bc_step": 0,
    "max_timesteps": 10000,
    "num_epochs": 1000,
    "updates_per_step": 50,
    "samplesteps": 1000,
    "train_agent_interval": 200
  }
}

