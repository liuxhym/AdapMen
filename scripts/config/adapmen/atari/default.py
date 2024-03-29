default_args ={
  "task_name": 'AdapMen',
  "algo": 'AdapMen',
  "log_interval": 1000,
  "eval_interval": 1000,
  "snapshot_interval": 50000,
  "buffer_size": 50000,
  "teacher": "dqn-btq",
  "horizon": 100,

  "sac":{
    "actor_network_params": [("conv2d", 16, 8, 4, 0), ("conv2d", 32, 4, 2, 0),("flatten",), ("mlp", 256), ("mlp", 256)]
  },
  "env":{
    "env_name" : '',
    "discount" : 0.99,
    "num_traj": 0,
    "expert_data_path": '',
    "use_normalize_states": False,
    "subsample_rate": 1.0,
  },
  "btq":{
    "insert_all": False,
    "bc_type": "mse",
    "criterion_type": "q",
    "start_training_timesteps": 0,
    "actor_lr" : 3e-4,
    "batch_size": 32,
    "bc_step": 0,
    "max_timesteps": 100000,
    "update_interval": 200,
    "updates_per_step": 200,
    "target_prob": 0.2,
    "update_p_freq": 200,
    "p_value": 0.5,
    "samplesteps": 2000,
    "smooth": 10,
  }
}